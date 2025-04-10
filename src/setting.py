import os

now_path = os.path.dirname(__file__)

# generate 폴더 경로
generate_save_path = os.path.join(now_path, '..', 'generate')
generate_save_path = os.path.abspath(generate_save_path)

# config 폴더 경로
config_path = os.path.join( now_path , '..', 'config')
config_path = os.path.abspath(config_path)

# docs 폴더 경로
docs_path = os.path.join( now_path , '..', 'docs')
docs_path = os.path.abspath(docs_path)