

import json
import requests
import pandas as pd

class Evaluation(object):
    def __init__(self, dataset_id=None, experiment_id=None):

        self.dataset_url='http://evaluation-internal.visenze.com/api/v1/recognition/dataset/{}/json'
        self.experiment_url='http://evaluation-internal.visenze.com/api/v1/recognition/experiment/{}/json'
        #todo: put username and password to confidential yaml
        self.username='admin'
        self.password='evaluation_123'
        self.DATASET_COLS=['im_url', 'public_url','concepts', 'box', 'im_cat',
                'dataset_id', 'dataset_name', 'im_source']
        self.dataset_id=dataset_id
        self.experiment_id=experiment_id

        if dataset_id==None and experiment_id==None:
            raise Exception('Either dataset ID or experiment ID must be provided.')
        elif dataset_id!=None and experiment_id!=None:
            raise Exception('Only dataset ID or experiement ID can be provided, but not both.')

    def get_data_dict(self):
        if self.dataset_id!=None:
            url=self.dataset_url
            data_id=self.dataset_id
        else:
            url=self.experiment_url
            data_id=self.experiment_id
        dataset=requests.get(url.format(data_id), auth=(self.username, self.password))
        dataset_dict=json.loads(dataset.text)['data']
        return dataset_dict

    def get_dataset_dataframe(self):
        dataset_dict=self.get_data_dict()
        dataset_df=pd.DataFrame(dataset_dict)
        dataset_df['concepts']=dataset_df.attribute+'/'+dataset_df.tag
        dataset_df['dataset_id']=self.dataset_id
        dataset_df['im_source']='evaluation_dataset'

        #remove ','in 'detection' columns
        dataset_df['detection']=dataset_df.detection.str.replace(',','')

        #todo: find a way to import dataset_name from evaluation system
        dataset_df['dataset_name']=''


        dataset_df=dataset_df[['testSetUrl', 's3Url', 'concepts', 'box', 'detection', 'dataset_id', 'dataset_name', 'im_source']]
        dataset_df.columns=self.DATASET_COLS
        return dataset_df


    def get_experiment_dataframe(self):
        def find_tag_group(groundTruths):
            if ':' in groundTruths:
                gt_lst=groundTruths.split(':')
                return gt_lst[-2]
            else:
                gt_lst=''
                return ''

        def find_index(tag_group, all_tag):
            #find the index of tag_group in the list of tags
            return [tag_group in element for element in all_tag].index(True)




        exp_dict=self.get_data_dict()
        exp_df=pd.DataFrame(exp_dict)

        return_df=pd.DataFrame({'im_url': exp_df.originUrl,
                               'public_url': exp_df.s3Url})
        return_df['box']=exp_df.groundTruths.apply(lambda x: x[0]['box'])
        return_df['groundTruths']=exp_df.groundTruths.apply(lambda x: x[0]['concept'])
        return_df['groundTruths_tagGroup']=return_df.groundTruths.apply(find_tag_group)
        return_df['all_pred_tag']=[[x['concept'] for x in y] for y in exp_df.predictions]
        return_df['all_pred_box']=[[x['box'] for x in y] for y in exp_df.predictions]
        return_df['all_pred_score']=[[x['score'] for x in y] for y in exp_df.predictions]

        #find predicted tag, box and score for each row
        return_df['predicted_tag']=''
        return_df['predicted_box']=''
        return_df['predicted_score']=''
        for i in range (len(return_df)):
            groundTruths_tagGroup=return_df.groundTruths_tagGroup.iloc[i]
            idx=find_index(groundTruths_tagGroup, return_df.all_pred_tag.iloc[i])
            return_df['predicted_tag'].iloc[i]=return_df.all_pred_tag.iloc[i][idx]
            return_df['predicted_box'].iloc[i]=return_df.all_pred_box.iloc[i][idx]
            return_df['predicted_score'].iloc[i]=return_df.all_pred_score.iloc[i][idx]
        #convert predicted_box column to hydra box format
        return_df.predicted_box=return_df.predicted_box.astype(str).str.replace('[', '').str.replace(']', '')
        return_df['debug']=exp_df.predictions

        return return_df.drop(['all_pred_tag', 'all_pred_box', 'all_pred_score'], axis=1)

#sample usage
print ('fetching evaluation data...')
eval_data=Evaluation(experiment_id=7709)
eval_dict=eval_data.get_data_dict()
print ('procecssing the result...')
eval_df=eval_data.get_experiment_dataframe()


print (eval_df.head())



