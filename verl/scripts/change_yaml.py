import argparse
import yaml

def load_yaml(file_path):
    """
    读取并解析 YAML 文件。
    :param file_path: YAML 文件的路径
    :return: 解析后的字典对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到！")
        exit(1)
    except yaml.YAMLError as e:
        print(f"错误：YAML 文件解析失败！\n{e}")
        exit(1)

def save_yaml(data, output_file):
    """
    将数据保存为 YAML 文件。
    :param data: 要保存的数据（字典）
    :param output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            yaml.dump(data, file, allow_unicode=True, sort_keys=False)
        print(f"已成功保存到 '{output_file}'")
    except Exception as e:
        print(f"错误：无法保存 YAML 文件！\n{e}")
        exit(1)

def modify_yaml(data, key, value):
    """
    修改 YAML 数据中的某个键的值。
    :param data: 原始 YAML 数据（字典）
    :param key: 要修改的键（支持嵌套键，用 '.' 分隔）
    :param value: 新的值
    """
    keys = key.split('.')
    current = data
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def parse_key_value_pairs(pairs):
    """
    解析键值对列表，并尝试推断值的类型。
    :param pairs: 键值对列表，格式为 ['key1=value1', 'key2=value2', ...]
    :return: 键值对字典
    """
    result = {}
    for pair in pairs:
        if '=' not in pair:
            print(f"错误：无效的键值对格式 '{pair}'。应为 'key=value'")
            exit(1)
        key, value = pair.split('=', 1)
        try:
            # 尝试将值解析为 YAML 支持的类型
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError:
            # 如果解析失败，则保留原始字符串
            parsed_value = value
        result[key] = parsed_value
    return result

def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="读取 YAML 文件，批量修改配置并保存为新文件。")
    
    # 添加命令行参数
    parser.add_argument(
        "-f", "--file", 
        required=True, 
        help="指定输入的 YAML 文件路径"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="指定输出的 YAML 文件路径"
    )
    parser.add_argument(
        "-kv", "--key-value", 
        nargs='+', 
        required=True, 
        help="要修改的键值对，格式为 'key1=value1 key2=value2 ...'"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载 YAML 文件
    yaml_data = load_yaml(args.file)
    
    # 解析键值对
    key_value_pairs = parse_key_value_pairs(args.key_value)
    
    # 批量修改 YAML 数据
    for key, value in key_value_pairs.items():
        modify_yaml(yaml_data, key, value)
    
    # 保存修改后的 YAML 数据
    save_yaml(yaml_data, args.output)

if __name__ == "__main__":
    main()