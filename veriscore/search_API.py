import os
from ast import literal_eval
from pathlib import Path
import json
from diskcache.core import Cache
import requests


class SearchAPI:
    def __init__(self):
        # invariant variables
        self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json",
        }
        # cache related
        self.cache_file = "data/cache/search_cache.json"
        self.cache_dict = self.load_cache()
        self.cache = self._load_cache("data/cache/diskcache", self.cache_file)
        self.add_n = 0
        self.save_interval = 10

    def _load_cache(self, cache_dir: str, old_cache_path: str) -> Cache:
        cache = Cache(cache_dir, eviction_policy="none", size_limit=2 ** 34)
        if Path(old_cache_path).is_file():
            cache_dict = json.loads(Path(old_cache_path).read_text())
            for k, v in cache_dict.items():
                cache.add(k, v)

        return cache

    def get_snippets(self, claim_lst):
        text_claim_snippets_dict = {}

        search_results = self.batch_search(claim_lst)

        for query in claim_lst:
            search_result = search_results[query]
            # search_result = self.get_search_res(query)
            if "statusCode" in search_result:  # and search_result['statusCode'] == 403:
                print(search_result["message"])
                exit()
            if "organic" in search_result:
                organic_res = search_result["organic"]
            else:
                organic_res = []

            search_res_lst = []
            for item in organic_res:
                title = item["title"] if "title" in item else ""
                snippet = item["snippet"] if "snippet" in item else ""
                link = item["link"] if "link" in item else ""

                search_res_lst.append(
                    {"title": title, "snippet": snippet, "link": link}
                )
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def _batch_search_and_cache(self, queries: list[str]) -> dict[str, dict]:
        results: dict[str, dict] = {
            query: self.cache.get(query.strip()) for query in queries
        }  # type: ignore
        uncached_queries = [query for query, res in results.items() if res is None]

        if len(uncached_queries) > 0:
            payload = json.dumps([{"q": query} for query in uncached_queries])
            response = requests.request(
                "POST", self.url, headers=self.headers, data=payload
            )
            response_batch = response.json()

            for res, query in zip(response_batch, uncached_queries):
                assert res["searchParameters"]["q"] == query.strip()
                self.cache.add(query.strip(), res)
                results[query] = res

        return results

    def batch_search(self, queries: list[str]) -> dict[str, dict]:
        results = {}

        MAX_BATCH_SIZE = 100
        for i in range(0, len(queries), MAX_BATCH_SIZE):
            res_dict = self._batch_search_and_cache(queries[i : i + MAX_BATCH_SIZE])
            results.update(res_dict)

        return results

    def get_search_res(self, query):
        # check if prompt is in cache; if so, return from cache
        cache_key = query.strip()
        res = self.cache.get(cache_key)
        if res is not None:
            return res
        # if cache_key in self.cache_dict:
        #     # print("Getting search results from cache ...")
        #     return self.cache_dict[cache_key]

        payload = json.dumps({"q": query})
        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload
        )
        response_json = literal_eval(response.text)

        # update cache
        self.cache.set(query.strip(), response_json)
        # self.cache_dict[query.strip()] = response_json
        self.add_n += 1

        # # save cache every save_interval times
        # if self.add_n % self.save_interval == 0:
        #     self.save_cache()

        return response_json

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        print("Saving search cache ...")
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                # load a json file
                cache = json.load(f)
                print("Loading cache ...")
        else:
            cache = {}
        return cache
