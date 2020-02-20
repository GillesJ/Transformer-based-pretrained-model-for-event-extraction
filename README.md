# Transformer-based-pretrained-model-for-event-extraction
Pre-trained language models such as bert / gpt2 / albert / xlm / roberta are used to perform event extraction tasks on ace2005. The code is modified on the nlpcl-lab / bert-event-extraction framework, replacing the model building part of the original project with the transformer and crf models of pytorch. The whole model adopts the sequence labeling method without using any auxiliary information. Use crf as the trigger word recognition, and then use crf for argument recognition based on the trigger word recognition results. When the pre-trained model selects xlm-roberta-large, trigger-f1 = 0.72; argument-f1 = 0.45. argument increased by 0.05.

[trigger classification]	P=0.677	R=0.754	F1=0.713
[argument classification]	P=0.588	R=0.384	F1=0.464
[trigger identification]	P=0.723	R=0.805	F1=0.762
[argument identification]	P=0.617	R=0.403	F1=0.488

The hyperparameters are as follows
===================== hyperparameters ====================== model
PreTrainModel = ['Bert_large', 'Gpt', 'Gpt2', 'Ctrl', 'TransfoXL', 'Xlnet_base', 'Xlnet_large', 'XLM', 'DistilBert_base', 'DistilBert_large', 'Roberta_base', 'Roberta_large', 'XLMRoberta_base', 'XLMRoberta_large', 'ALBERT-base-v1', 'ALBERT-large-v1', 'ALBERT-xlarge-v1', 'ALBERT-xxlarge-v1', 'ALBERT-base-v2', 'ALBERT-large-v2', 'ALBERT-xlarge-v2', 'ALBERT-xxlarge-v2']
           early_stop = 5
                   lr = 1e-05
                   l2 = 1e-05
             n_epochs = 50
               logdir = logdir
             trainset = data/train_balance.json
               devset = data/dev.json
              testset = data/test.json
           LOSS_alpha = 1.0
   telegram_bot_token = 
     telegram_chat_id = 
       PreTrain_Model = XLMRoberta_large
           model_path = /content/drive/My Drive/Colab Notebooks/model/eventextraction/Transformer-based-pretrained-model-for-event-extraction-master/save_model/latest_model.pt
           batch_size = 16
