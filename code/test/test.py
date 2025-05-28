from FlagEmbedding import FlagModel

model_klue = FlagModel("klue/roberta-large")

model_baai = FlagModel("BAAI/bge-large-en-v1.5")

model_raw = FlagModel("/data2/local_datasets/encoder/bgem3")

model_custom = FlagModel(
    model_name_or_path="/data2/local_datasets/encoder/bgem3_custom",
)

model_raw_ft = FlagModel("/data2/local_datasets/encoder/bgem3/ft")

model_custom_ft = FlagModel(
    model_name_or_path="/data2/local_datasets/encoder/bgem3_custom/ft",
)

queries = [
    "하염없이 흐르는 비",
    "그대로 눈이 되어 내려라 오 내려라",
    "비가 온다 눈이 되지 못한 채",
]


q_baai = model_baai.encode_queries(queries)

# 밑에 3개 결과같음

q_klue = model_klue.encode_queries(queries)  # 그냥 bge m3 의 backbone

q_raw = model_raw.encode_queries(queries)  # bge m3

q_custom = model_custom.encode_queries(queries)  # 커스텀한 bge m3

# 밑에 2개 결과같음

q_raw_ft = model_raw_ft.encode_queries(queries)

q_custom_ft = model_custom_ft.encode_queries(queries)


print(q_baai)
print()
#

print(q_klue)
print()
print(q_raw)
print()
print(q_custom)
print()
#
print("## FT ##")
print(q_raw_ft)
print()
print(q_custom_ft)
