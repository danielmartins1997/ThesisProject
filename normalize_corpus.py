import pandas as pd
import numpy as np

# Takes a CSV file as input and performs some data preprocessing on it. return the dataframe corpus preprocessed
def normalizeDataset(csvFileInput):
	df_initial = pd.read_csv(csvFileInput, on_bad_lines='skip', sep=";")
	
	# Renames columns in the DataFrame to make them consistent
	if 'text' in df_initial.columns:
		df_initial.rename(columns={'text': 'utterance'}, inplace=True)
	if 'Utterance' in df_initial.columns:
		df_initial.rename(columns={'Utterance': 'utterance'}, inplace=True)
	if 'transcript' in df_initial.columns:
		df_initial.rename(columns={'transcript': 'utterance'}, inplace=True)
	if 'intent_title' in df_initial.columns:
		df_initial.rename(columns={'intent_title': 'trueLabel'}, inplace=True)
	if 'intents' in df_initial.columns:
		df_initial.rename(columns={'intents': 'trueLabel'}, inplace=True)
	if 'active_intent' in df_initial.columns:
		df_initial.rename(columns={'active_intent': 'trueLabel'}, inplace=True)
	if 'intent' in df_initial.columns:
		df_initial.rename(columns={'intent': 'trueLabel'}, inplace=True)
	if 'dialog_act' in df_initial.columns:
		df_initial.rename(columns={'dialog_act': 'trueLabel'}, inplace=True)
	if 'dialogue_act' in df_initial.columns:
		df_initial.rename(columns={'dialogue_act': 'trueLabel'}, inplace=True)
	if 'utterance' in df_initial.columns:
		df_initial['utterance'] = df_initial['utterance'].astype(str)

	df_final = df_initial[["utterance", "trueLabel"]]
	return df_final