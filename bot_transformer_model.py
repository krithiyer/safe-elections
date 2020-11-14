
from abc import ABC
import json
import logging
import os
import pandas
import torch

from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline
from keras.preprocessing.sequence import pad_sequences


class BotQuestionClassifier(ABC):

    def __init__ (self, req_match_questions = 3, faq_db_name= './final_database.xlsx', model_dir_name='./QQP_finetuned/'):
        self.faq_file_name = faq_db_name
        self.model_dir = model_dir_name
        self.TOKEN_MAX_LEN = 256
        self.num_selected_ans = req_match_questions # number of question to question match
        election_faq_df = pandas.read_excel(self.faq_file_name)
        self.master_faq_questions_df = election_faq_df.dropna()
        self.model = BertForSequenceClassification.from_pretrained(self.model_dir, output_hidden_states = True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.qa_pipeline = pipeline("question-answering")
        #self.device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

    # read the faq questions and answers    
    def load_faqData (self):
        election_faq_df = pandas.read_excel(self.faqFileName)
        self.master_faq_questions_df = election_faq_df.dropna()
        return

    # tokenize user_query and question from faq_database
    def create_tokens (self, user_query, faq_question):
        input_ids = self.tokenizer.encode(
                        user_query, faq_question,                   # Sentence to encode.
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        max_length = self.TOKEN_MAX_LEN,       # Truncate all sentences. 
                        truncation = True                       
                   )  
        results = pad_sequences([input_ids], maxlen=self.TOKEN_MAX_LEN, dtype="long", 
                              truncating="post", padding="post")  
        # Remove the outer list.
        input_ids = results[0]
        # Create attention masks    
        attn_mask = [int(i>0) for i in input_ids]
        # Cast to tensors.
        input_ids = torch.tensor(input_ids)
        attn_mask = torch.tensor(attn_mask)
        # Add an extra dimension for the "batch" (even though there is only one 
        # input in this batch.)
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)  

        return input_ids, attn_mask


    def find_matching_questions (self, userQuery):
        query_match_qQA_df = pandas.DataFrame(columns = ['User_Question', 'FAQ_Question', 'FAQ_Answer','Logit_0', 'Logit_1', 'Probability_0', 'Probability_1'])
        tempResults_eachQ_df = pandas.DataFrame(columns = ['User_Question', 'FAQ_Question', 'FAQ_Answer','Logit_0', 'Logit_1', 'Probability_0', 'Probability_1'])
        for row in self.master_faq_questions_df.itertuples():
            inpt_id, att_mask = self.create_tokens(userQuery, row.Question)
            inpt_id = inpt_id.to(self.device)
            att_mask = att_mask.to(self.device)
            #logits.cuda()
            #encoded_layers.cuda()
            with torch.no_grad():
                logits, encoded_layers = self.model(input_ids = inpt_id, token_type_ids = None, attention_mask = att_mask)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            sclr_logits = logits.cpu().numpy()
            sclr_prob = probabilities.cpu().numpy()
            temp_df = pandas.DataFrame( [[ userQuery, row.Question, row.Answer, sclr_logits[0][0], sclr_logits[0][1], sclr_prob[0][0], sclr_prob[0][1] ] ],
                     columns = ['User_Question', 'FAQ_Question', 'FAQ_Answer','Logit_0', 'Logit_1', 'Probability_0', 'Probability_1'])
            tempResults_eachQ_df = tempResults_eachQ_df.append(temp_df)
        sorted_tempResults_df = tempResults_eachQ_df.sort_values(by=['Probability_1'], ascending=False)
        query_match_qQA_df = query_match_qQA_df.append(sorted_tempResults_df.head(self.num_selected_ans))
        return query_match_qQA_df

    def find_matching_answers (self, userQuery, selectedQuestions_df):
        response_final_match_df = pandas.DataFrame(columns = ['User_Question', 'FAQ_Question', 'FAQ_Answer','Matched_Answer', 'Match_Score'])
        for row in selectedQuestions_df.itertuples():
            result = self.qa_pipeline(question=row.User_Question, context=row.FAQ_Answer)
            temp_df = pandas.DataFrame( [[ row.User_Question, row.FAQ_Question, row.FAQ_Answer, result['answer'], round(result['score'], 4) ] ],
                     columns = ['User_Question', 'FAQ_Question', 'FAQ_Answer','Matched_Answer', 'Match_Score'])
            response_final_match_df = response_final_match_df.append(temp_df)
            response_final_match_df = response_final_match_df.sort_values(by=['Match_Score'], ascending=False)
        return response_final_match_df


