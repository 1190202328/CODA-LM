import argparse
import json
import os.path
import re


def analyse(directory, num_samples=10):
    score_dict = {
        'region_perception': [],
        'general_perception': [],
        'driving_suggestion': []
    }

    for perception in score_dict:
        pred_path = f'{directory}/{perception}_answer'
        if perception == 'region_perception':
            pred_path = f'{pred_path}_w_label'
        pred_path = f'{pred_path}.jsonl'
        with open(pred_path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                json_line = json.loads(line)
                info_dict = {
                    'image': f"{json_line['image']}",
                    'pred': json_line['answer']
                }
                if 'label_name' in json_line:
                    info_dict['label_name'] = json_line['label_name']
                score_dict[perception].append(info_dict)

        answer_path = f'{answer_root}/{perception}.jsonl'
        with open(answer_path, mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                json_line = json.loads(line)
                score_dict[perception][i]['ground_truth'] = json_line['answer']

                image_name = score_dict[perception][i]['image']
                split = image_name.split('/')[0]
                image_name = image_name.split('/')[-1].split('.')[0]
                image_name = f'{split}_{image_name}'

                judge_file = f'{directory}/{perception}_answer'
                if perception == 'region_perception':
                    judge_file = f'{judge_file}/gpt_result'
                judge_file = f'{judge_file}/{image_name}.txt'
                if not os.path.exists(judge_file):
                    score_dict[perception][i]['rate'] = 0
                    continue
                with open(judge_file, mode='r', encoding='utf-8') as judge_f:
                    judge_txt = judge_f.read()
                matches = re.findall('\[\[.*\]\]', judge_txt)
                if len(matches) != 1:
                    raise Exception
                rate = float(matches[0][2:-2])

                score_dict[perception][i]['judge'] = judge_txt
                score_dict[perception][i]['rate'] = rate

    for perception in score_dict:
        score_dict[perception] = sorted(score_dict[perception], key=lambda x: x['rate'])

    for perception in score_dict:
        print('-' * 30 + f'{perception}' + '-' * 30)
        for i, item in enumerate(score_dict[perception]):
            if i >= num_samples:
                break
            print(item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    answer_root = '/Users/didi/Desktop/ECCV比赛/验证数据保存/NEW_Mini/vqa_anno'
    image_root_dir = '/nfs/ofs-902-1/object-detection/tangwenbo/vlm/data/'

    print(image_root_dir)
    analyse(directory=args.directory, num_samples=args.num_samples)
