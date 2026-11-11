import os
import glob
import csv
from tensorboard.backend.event_processing import event_accumulator

# 你可以根据实际情况修改这些tag
SCALAR_TAGS = [
    'loss',
    'Recall@1',
    'Recall@5',
    'Recall@10',
    'rsum',
    'val/loss',
    'val/Recall@1',
    'val/Recall@5',
    'val/Recall@10',
    'val/rsum',
]

def extract_scalars_from_event(event_path, tags=SCALAR_TAGS):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    scalars = {}
    for tag in tags:
        if tag in ea.Tags().get('scalars', []):
            scalars[tag] = ea.Scalars(tag)
    return scalars

def main():
    log_dir = 'logs'
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print('未找到任何 TensorBoard 事件文件！')
        return
    for event_file in event_files:
        print(f'正在处理: {event_file}')
        scalars = extract_scalars_from_event(event_file)
        if not scalars:
            print('  没有找到指定的标量数据。')
            continue
        # 以事件文件名命名输出csv
        csv_name = os.path.splitext(os.path.basename(event_file))[0] + '.csv'
        csv_path = os.path.join(log_dir, csv_name)
        # 收集所有step
        all_steps = set()
        for v in scalars.values():
            all_steps.update([x.step for x in v])
        all_steps = sorted(all_steps)
        # 写入csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['step'] + list(scalars.keys())
            writer.writerow(header)
            for step in all_steps:
                row = [step]
                for tag in scalars.keys():
                    # 找到该step对应的值
                    value = next((x.value for x in scalars[tag] if x.step == step), '')
                    row.append(value)
                writer.writerow(row)
        print(f'  已导出到: {csv_path}')

if __name__ == '__main__':
    main() 