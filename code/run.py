import datetime
import logging
from PLM import PLM_model
from utils import set_seed
import pandas as pd
import tqdm

set_seed(42)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'codet5': 'Salesforce/codet5-base'
}

model_type = 'codet5'

# init
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type],
                  load_model_path=None,
                  beam_size=10, max_source_length=512, max_target_length=64)

start = datetime.datetime.now()

lans = ['python', 'java', 'javascript', 'c#', 'php', 'ruby', 'go', 'html']
# train
train_filename = 'train.csv'
eval_filename = 'valid.csv'
model.train(train_filename=train_filename, train_batch_size=8, learning_rate=5e-5,
            num_train_epochs=30, early_stop=3, do_eval=True, eval_filename=eval_filename,
            eval_batch_size=8, output_dir='valid_output/codet5', do_eval_bleu=True, save_steps=6000)
end = datetime.datetime.now()
print(f"train: {end - start}")

# reload
start = datetime.datetime.now()
for lan in lans:
    model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10,
                      max_source_length=512, max_target_length=64,
                      load_model_path='valid_output/codet5/checkpoint-best-bleu/pytorch_model.bin')
    filename = f'./{lan}./test.csv'
    output_dir = f'../result/codet5/' + lan + '/'
    model.test(batch_size=8, filename=filename, output_dir=output_dir)
end = datetime.datetime.now()
print(f"test: {end - start}")
