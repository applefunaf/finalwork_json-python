# import requirements
import json
import os
from ord_schema.message_helpers import load_message, write_message
from ord_schema.proto import dataset_pb2
from google.protobuf.json_format import MessageToJson

input_dir = "data"
output_dir = "data_json"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".pb.gz"):
            input_fname = os.path.join(root, filename)
            dataset = load_message(
                input_fname,
                dataset_pb2.Dataset,
            )
            
            for i, rxn in enumerate(dataset.reactions):
                rxn_json = json.loads(
                    MessageToJson(
                        message=rxn,
                        including_default_value_fields=False,
                        preserving_proto_field_name=True,
                        indent=2,
                        sort_keys=False,
                        use_integers_for_enums=False,
                        descriptor_pool=None,
                        float_precision=None,
                        ensure_ascii=True,
                    )
                )
                
                # 保持输出文件夹结构
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                output_fname = os.path.join(output_subdir, f"{os.path.splitext(filename)[0]}_reaction_{i}.json")
                with open(output_fname, "w", encoding="utf-8") as f:
                    json.dump(rxn_json, f, ensure_ascii=False, indent=2)
                print(f"JSON data has been written to {output_fname}")