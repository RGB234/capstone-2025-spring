from safetensors.torch import load_file

# safetensors 파일 경로
ft_filepath = "/data2/local_datasets/encoder/bgem3_custom/ft/model.safetensors"

# state_dict 로드
ft_state_dict = load_file(ft_filepath)

# 키와 shape 출력

print("## FT model ##")
for key, value in ft_state_dict.items():
    # print(f"{key}: \n{value}\n")
    if key == "encoder.layer.23.attention.output.dense.weight":
        print(f"{key}: {value}")

##########################

raw_ft_filepath = "/data2/local_datasets/encoder/bgem3/ft/model.safetensors"

raw_ft_state_dict = load_file(raw_ft_filepath)

print("## Raw FT model ##")

for key, value in raw_ft_state_dict.items():
    if key == "encoder.layer.23.attention.output.dense.weight":
        print(f"{key}: {value}")
