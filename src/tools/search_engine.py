# -*- coding: utf-8 -*-
"""
tools/search_engine.py — 无 API Key 的网页搜索引擎

基于页面抓取实现百度 / Bing 国际版搜索，不依赖额外搜索平台聚合器。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import quote

import requests
from lxml import html


class BaseSearchEngine(ABC):
    """无需 API Key 的搜索引擎基类。"""

    ENGINE_NAME = ""
    ENGINE_DISPLAY_NAME = ""
    SEARCH_URL = ""

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

    @abstractmethod
    def build_url(self, query: str) -> str:
        """构造搜索 URL。"""

    @abstractmethod
    def parse(self, tree: html.HtmlElement) -> list[dict[str, Any]]:
        """从搜索结果页中抽取结构化结果。"""

    def search(
        self,
        query: str,
        max_results: int = 10,
        timeout: int = 15,
    ) -> list[dict[str, Any]]:
        url = self.build_url(query)
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = "utf-8"
            tree = html.fromstring(response.text)
            return self.parse(tree)[:max_results]
        except Exception as exc:
            raise RuntimeError(f"{self.ENGINE_DISPLAY_NAME} 搜索失败: {exc}") from exc


class BaiduEngine(BaseSearchEngine):
    """百度搜索。"""

    ENGINE_NAME = "baidu"
    ENGINE_DISPLAY_NAME = "百度"
    SEARCH_URL = "https://www.baidu.com/s?wd={q}"

    def build_url(self, query: str) -> str:
        return self.SEARCH_URL.format(q=quote(query))

    def parse(self, tree: html.HtmlElement) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        items = tree.xpath(
            "//div[contains(@class,'result')] | "
            "//div[contains(@class,'c-container')]"
        )
        for item in items:
            try:
                title = "".join(
                    item.xpath(
                        ".//h3//a//text() | "
                        ".//a[@class='c-title']//text()"
                    )
                ).strip()
                href = "".join(
                    item.xpath(
                        ".//h3//a/@href | "
                        ".//a[@class='c-title']/@href"
                    )
                ).strip()
                body = "".join(
                    item.xpath(
                        ".//div[contains(@class,'c-abstract')]//text() | "
                        ".//div[contains(@class,'abstract')]//text()"
                    )
                ).strip()
                if title and href:
                    results.append({
                        "title": title,
                        "href": href,
                        "body": body,
                        "source": self.ENGINE_NAME,
                    })
            except Exception:
                continue
        return results


class BingIntlEngine(BaseSearchEngine):
    """Bing 国际版搜索。"""

    ENGINE_NAME = "bing"
    ENGINE_DISPLAY_NAME = "Bing 国际"
    SEARCH_URL = "https://www.bing.com/search?q={q}"

    def build_url(self, query: str) -> str:
        return self.SEARCH_URL.format(q=quote(query))

    def parse(self, tree: html.HtmlElement) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        items = tree.xpath("//li[contains(@class,'b_algo')]")
        for item in items:
            try:
                title = "".join(item.xpath(".//h2//a//text()")).strip()
                href = "".join(item.xpath(".//h2//a/@href")).strip()
                body = "".join(
                    item.xpath(".//div[contains(@class,'b_caption')]//p//text()")
                ).strip()
                if title and href:
                    results.append({
                        "title": title,
                        "href": href,
                        "body": body,
                        "source": self.ENGINE_NAME,
                    })
            except Exception:
                continue
        return results


ENGINES: dict[str, type[BaseSearchEngine]] = {
    "baidu": BaiduEngine,
    "bing": BingIntlEngine,
}

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def choose_engine(query: str) -> str:
    """简单按查询语言自动选引擎。"""
    return "baidu" if _ZH_RE.search(query or "") else "bing"


def create_engine(engine: str, proxy: str | None = None) -> BaseSearchEngine:
    if engine == "auto":
        raise ValueError("create_engine 不接受 auto，请先调用 choose_engine()")
    if engine not in ENGINES:
        raise ValueError(f"未知搜索引擎: {engine}")
    return ENGINES[engine](proxy=proxy)


def run_search(
    engine: str,
    query: str,
    max_results: int = 10,
    proxy: str | None = None,
) -> list[dict[str, Any]]:
    actual_engine = choose_engine(query) if engine == "auto" else engine
    inst = create_engine(actual_engine, proxy=proxy)
    return inst.search(query, max_results=max_results)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="无需 API Key 的网页搜索工具（百度 / Bing 国际）"
    )
    parser.add_argument("engine", choices=["auto", *ENGINES.keys()], help="搜索引擎")
    parser.add_argument("query", help="搜索关键词")
    parser.add_argument("-n", "--max-results", type=int, default=10, help="最大结果数")
    parser.add_argument("--proxy", help="代理地址，例如 http://127.0.0.1:7890")
    parser.add_argument(
        "--json",
        dest="fmt",
        action="store_const",
        const="json",
        default="text",
        help="以 JSON 格式输出",
    )
    args = parser.parse_args()

    try:
        results = run_search(
            engine=args.engine,
            query=args.query,
            max_results=args.max_results,
            proxy=args.proxy,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.fmt == "json":
        print(json.dumps({
            "engine": args.engine,
            "query": args.query,
            "results": results,
        }, ensure_ascii=False, indent=2))
        return 0

    actual_engine = choose_engine(args.query) if args.engine == "auto" else args.engine
    print(f"🔍 {actual_engine} › {args.query}  ({len(results)} 条)\n")
    for idx, result in enumerate(results, 1):
        print(f"[{idx}] {result['title']}")
        print(f"    🔗 {result['href']}")
        if result.get("body"):
            print(f"    📝 {result['body'][:120]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
