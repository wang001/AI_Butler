# -*- coding: utf-8 -*-
"""
tools/web_fetcher.py — 轻量网页正文抓取工具

实现思路参考 shirenchuang/web-content-fetcher：
  - 直接通过 HTTP 抓取页面
  - 按常见正文容器优先级提取主要内容
  - 修复常见懒加载图片属性
  - 输出 markdown 或纯文本

定位：
  - 适合“已知 URL，想抓网页正文”的场景
  - 不执行 JavaScript，不处理登录态
  - 若页面强依赖 JS / 交互，应改用 browser_use
"""
from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from lxml import html


WEB_FETCHER_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_fetcher",
            "description": (
                "抓取指定网页 URL 的正文内容。"
                "适用于用户已经提供具体网页链接，想提取文章正文、公告内容、博客内容、文档页面内容等场景。"
                "默认输出 markdown；如果只要纯文本，可设为 text。"
                "此工具不执行 JavaScript，也不处理登录态。遇到强依赖 JS 的页面，请改用 browser_use。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页 URL，必须是 http 或 https。",
                    },
                    "extract_mode": {
                        "type": "string",
                        "enum": ["markdown", "text"],
                        "default": "markdown",
                        "description": "正文输出格式，默认 markdown。",
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 12000,
                        "minimum": 1000,
                        "maximum": 50000,
                        "description": "正文最大返回字符数，默认 12000。",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 5,
                        "maximum": 60,
                        "description": "HTTP 请求超时秒数，默认 20。",
                    },
                },
                "required": ["url"],
            },
        },
    },
]


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_BLOCK_TAGS = {
    "article", "main", "section", "div", "p", "ul", "ol", "li", "pre",
    "blockquote", "table", "thead", "tbody", "tfoot", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6",
}
_REMOVE_XPATH = (
    ".//script | .//style | .//noscript | .//iframe | .//svg | .//canvas | "
    ".//*[contains(@class,'advert')] | .//*[contains(@class,'ads')] | "
    ".//*[contains(@class,'footer')] | .//*[contains(@class,'header')] | "
    ".//*[contains(@class,'sidebar')] | .//*[contains(@class,'comment')] | "
    ".//*[contains(@id,'comment')] | .//*[contains(@class,'related')]"
)


def _class_xpath(*class_names: str) -> str:
    conditions = " and ".join(
        f"contains(concat(' ', normalize-space(@class), ' '), ' {name} ')"
        for name in class_names
    )
    return f"//*[{conditions}]"


_CONTENT_XPATHS: list[tuple[str, str]] = [
    ("wechat", "//*[@id='js_content']"),
    ("article", "//article"),
    ("main", "//main"),
    ("post-content", _class_xpath("post-content")),
    ("entry-content", _class_xpath("entry-content")),
    ("article-content", _class_xpath("article-content")),
    ("content", _class_xpath("content")),
    ("post", _class_xpath("post")),
    ("entry", _class_xpath("entry")),
    ("markdown-body", _class_xpath("markdown-body")),
    ("doc-body", _class_xpath("doc-body")),
    ("role-main", "//*[@role='main']"),
    ("body", "//body"),
]


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_public_http_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "仅支持 http/https URL。"
    if not parsed.netloc:
        return False, "URL 缺少主机名。"
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False, "不允许抓取本地回环地址。"
    if host.endswith(".local"):
        return False, "不允许抓取本地域名。"
    return True, ""


def _clone_element(el: html.HtmlElement) -> html.HtmlElement:
    return html.fromstring(html.tostring(el, encoding="unicode", method="html"))


def _fix_lazy_images(root: html.HtmlElement, base_url: str) -> None:
    for img in root.xpath(".//img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or img.get("data-original-src")
            or img.get("data-actualsrc")
        )
        if src:
            img.set("src", urljoin(base_url, src))
        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            img.set("srcset", srcset)


def _pick_content_node(tree: html.HtmlElement) -> tuple[html.HtmlElement, str]:
    for name, xpath in _CONTENT_XPATHS:
        nodes = tree.xpath(xpath)
        for node in nodes:
            if not isinstance(node, html.HtmlElement):
                continue
            text_len = len(_normalize_text(node.text_content()))
            if text_len >= 120:
                return node, name
    body = tree.xpath("//body")
    if body:
        return body[0], "body"
    return tree, "document"


def _clean_content(root: html.HtmlElement) -> None:
    for node in root.xpath(_REMOVE_XPATH):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)


def _render_inline(node: html.HtmlElement, base_url: str) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)

    for child in node:
        if not isinstance(child.tag, str):
            continue
        parts.append(_render_node(child, base_url, inline=True))
        if child.tail:
            parts.append(child.tail)

    text = "".join(parts)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _render_list(items: Iterable[html.HtmlElement], base_url: str, ordered: bool) -> str:
    lines: list[str] = []
    for idx, item in enumerate(items, 1):
        content = _normalize_text(_render_inline(item, base_url))
        if not content:
            content = _normalize_text("".join(
                _render_node(child, base_url, inline=False) for child in item
                if isinstance(child.tag, str)
            ))
        if not content:
            continue
        prefix = f"{idx}. " if ordered else "- "
        lines.append(prefix + content.replace("\n", "\n  "))
    return "\n".join(lines) + ("\n\n" if lines else "")


def _render_node(node: html.HtmlElement, base_url: str, inline: bool = False) -> str:
    tag = node.tag.lower() if isinstance(node.tag, str) else ""
    if tag in {"script", "style", "noscript", "iframe", "svg", "canvas"}:
        return ""

    if tag == "br":
        return "\n"
    if tag == "a":
        text = _normalize_text(_render_inline(node, base_url)) or (node.get("href") or "")
        href = node.get("href")
        if href:
            href = urljoin(base_url, href)
            return f"[{text}]({href})"
        return text
    if tag == "img":
        src = node.get("src")
        alt = _normalize_text(node.get("alt") or "")
        if src:
            return f"![{alt}]({src})"
        return ""
    if tag in {"strong", "b"}:
        text = _normalize_text(_render_inline(node, base_url))
        return f"**{text}**" if text else ""
    if tag in {"em", "i"}:
        text = _normalize_text(_render_inline(node, base_url))
        return f"*{text}*" if text else ""
    if tag == "code":
        text = _normalize_text(_render_inline(node, base_url))
        return f"`{text}`" if text else ""
    if tag == "pre":
        code = node.text_content().rstrip()
        return f"```\n{code}\n```\n\n" if code else ""
    if tag == "blockquote":
        body = _normalize_text("".join(
            _render_node(child, base_url, inline=False) for child in node
            if isinstance(child.tag, str)
        ) or node.text_content())
        if not body:
            return ""
        return "\n".join(f"> {line}" if line else ">" for line in body.splitlines()) + "\n\n"
    if tag in {"ul", "ol"}:
        return _render_list(
            [child for child in node if isinstance(child.tag, str) and child.tag.lower() == "li"],
            base_url,
            ordered=(tag == "ol"),
        )
    if tag == "table":
        rows = []
        for row in node.xpath(".//tr"):
            cols = [
                _normalize_text("".join(col.itertext()))
                for col in row.xpath("./th|./td")
            ]
            cols = [col for col in cols if col]
            if cols:
                rows.append(" | ".join(cols))
        return "\n".join(rows) + ("\n\n" if rows else "")
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(tag[1])
        text = _normalize_text(_render_inline(node, base_url))
        return f"{'#' * level} {text}\n\n" if text else ""
    if tag == "p":
        text = _normalize_text(_render_inline(node, base_url))
        return f"{text}\n\n" if text else ""
    if tag == "li":
        text = _normalize_text(_render_inline(node, base_url))
        return f"- {text}\n" if text else ""

    rendered_children = "".join(
        _render_node(child, base_url, inline=(tag not in _BLOCK_TAGS))
        for child in node
        if isinstance(child.tag, str)
    )
    own_text = _normalize_text(node.text or "")

    if inline or tag not in _BLOCK_TAGS:
        return own_text + rendered_children

    combined = _normalize_text(own_text + rendered_children)
    if combined:
        return combined + "\n\n"
    return rendered_children


def _to_markdown(root: html.HtmlElement, base_url: str) -> str:
    parts: list[str] = []
    if root.text and root.tag.lower() not in _BLOCK_TAGS:
        parts.append(root.text)
    for child in root:
        if not isinstance(child.tag, str):
            continue
        parts.append(_render_node(child, base_url))
        if child.tail:
            parts.append(child.tail)
    if not parts:
        parts.append(_render_node(root, base_url))
    text = "".join(parts)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _to_text(root: html.HtmlElement) -> str:
    lines = []
    for chunk in root.text_content().splitlines():
        chunk = _normalize_text(chunk)
        if chunk:
            lines.append(chunk)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def web_fetcher(
    url: str,
    extract_mode: str = "markdown",
    max_chars: int = 12000,
    timeout: int = 20,
) -> str:
    ok, reason = _is_public_http_url(url)
    if not ok:
        return f"[web_fetcher 错误] {reason}"

    extract_mode = (extract_mode or "markdown").lower()
    if extract_mode not in {"markdown", "text"}:
        return "[web_fetcher 错误] extract_mode 仅支持 markdown 或 text。"

    max_chars = max(1000, min(max_chars, 50000))
    timeout = max(5, min(timeout, 60))

    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })

    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"[web_fetcher 失败] 请求失败：{exc}"

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return (
            "[web_fetcher 失败] 目标资源不是 HTML 页面。\n"
            f"URL：{response.url}\n"
            f"Content-Type：{response.headers.get('Content-Type', 'unknown')}"
        )

    response.encoding = response.encoding or response.apparent_encoding or "utf-8"

    try:
        tree = html.fromstring(response.text)
    except Exception as exc:
        return f"[web_fetcher 失败] HTML 解析失败：{exc}"

    title = _normalize_text("".join(tree.xpath("//title//text()")))
    content_node, selector_name = _pick_content_node(tree)
    content_root = _clone_element(content_node)
    _clean_content(content_root)
    _fix_lazy_images(content_root, response.url)

    if extract_mode == "markdown":
        content = _to_markdown(content_root, response.url)
    else:
        content = _to_text(content_root)

    if not content:
        return (
            "[web_fetcher 失败] 未提取到有效正文。\n"
            f"URL：{response.url}\n"
            f"标题：{title or '（无标题）'}"
        )

    truncated = False
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "\n\n[内容已截断]"
        truncated = True

    lines = [
        "[网页抓取结果]",
        f"URL：{response.url}",
        f"标题：{title or '（无标题）'}",
        f"提取模式：{extract_mode}",
        f"正文选择器：{selector_name}",
    ]
    if truncated:
        lines.append(f"最大字符数：{max_chars}")
    lines.extend(["", content])
    return "\n".join(lines)
