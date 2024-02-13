#!/usr/bin/env python3

""" Утилита для конвертации датасетов из https://github.com/Koziev/NLP_Datasets в подходящий для обучения моделей CDR формат.
"""

import json

# TODO: Переписать.

entries_count: int = 5000

data: list[dict[str, str]] = []

with open("../../cornell_movie_corpus.txt") as f:
  lines: list[str] = f.readlines()
  testX: list[str] = "".join(lines).split("\n\n\n\n")
  for testv in testX:
    testf: list[str] = list(filter(None, testv.split("\n")))
    # print(f"testf: {testf}")
    # print(f"testf len: {len(testf)}")
    i: int = 0
    while i <= len(testf)-1:
      i += 1

      if i+1 <= len(testf):
        query: str = testf[i - 1].replace("- ", "", 1).replace("--", "").strip()
        response: str = testf[i].replace("- ", "", 1).replace("--", "").strip()

        # print(">>>")
        # print(query)
        # print(response)
        # print("<<<")

        # max_data_size: int = 1000
        #
        # while len(data) < max_data_size:

        data.append({
          "query": query,
          "response": response
        })

  data: list[dict[str, str]] = data[:entries_count]

  # print(data)
  json.dump(data, open("../../cdr_data_lite.json", "w"), indent=2, ensure_ascii=False)
