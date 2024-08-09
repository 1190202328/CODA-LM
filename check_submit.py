import argparse
import json


def check(input_dir, ref='/Users/didi/Desktop/提交结果备份/ref/results'):
    json_files = ['driving_suggestion_answer.jsonl', 'general_perception_answer.jsonl',
                  'region_perception_answer.jsonl']

    ref_json_lists = []
    for name in json_files:
        path = f'{ref}/{name}'
        with open(path, encoding='utf-8', mode='r') as f:
            ref_json_lists.append(f.read().strip().split('\n'))

    for i, name in enumerate(json_files):
        path = f'{input_dir}/{name}'
        with open(path, encoding='utf-8', mode='r') as f:
            for j, line in enumerate(f.readlines()):
                line_json = json.loads(line)
                assert (line_json['answer'] is not None and len(line_json['answer']) != 0 and line_json[
                    'answer'] != 'None')
                ref_json = json.loads(ref_json_lists[i][j])
                assert line_json['answer'] != ref_json['answer']
                line_json['answer'] = ref_json['answer']
                assert line_json == ref_json
    print('done, no problem!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    args = parser.parse_args()
    check(args.directory)
