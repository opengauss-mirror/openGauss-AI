from pathlib import Path


base_dir = rath().resolve()
config = {}

table_job = {'aka_name': [], 'aka_title': [], 'cast_info': [], 'char_name': [], 'comp_cast_type': [], 'company_name': [], 'company_type': [], 'complete_cast': [], 'info_type': [], 'keyword': [], 'kind_type': [], 'link_type': [], 'movie_companies': [], 'movie_info': [], 'movie_info_idx': [], 'movie_keyword': [], 'movie_link': [], 'name': [], 'person_info': [], 'role_type': [], 'title': []}


def get_file(file_path, pattern="*"):
    """
    函数 获取给定目录下的所有文件的绝对路径
    参数 file_path: 文件目录
    参数 pattern:默认返回所有文件，也可以自定义返回文件类型，例如：pattern="*.py"
    返回值 abspath:文件路径列表
    """
    all_file = []
    files = Path(file_path).rglob(pattern)
    for file in files:
        if Path.is_file(file):
            all_file.append(file)
    return all_file



'''
BASE_DIR = Path('pybert')
config = {
    'raw_data_path': BASE_DIR / 'dataset/train.csv',
    'test_path': BASE_DIR / 'dataset/test_stage1.csv',

    'data_dir': BASE_DIR / 'dataset',
    'log_dir': BASE_DIR / 'output/log',
    'writer_dir': BASE_DIR / "output/TSboard",
    'figure_dir': BASE_DIR / "output/figure",
    'checkpoint_dir': BASE_DIR / "output/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",

    'bert_vocab_path': BASE_DIR / 'pretrain/bert/base-chinese/vocab.txt',
    'bert_config_file': BASE_DIR / 'pretrain/bert/base-chinese/config.json',
    'bert_model_dir': BASE_DIR / 'pretrain/bert/base-chinese',

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased'
}
'''