import pandas as pd

df = pd.read_csv("spine_raw.csv", header=None)
df = df.dropna(how="all")
df = df.reset_index(drop=True)
ls = ['LS']*41
lsac= ['AC']*41
lsas= ['AS']*41
lsfe= ['FE']*41
lslb= ['LB']*41
lsar= ['AR']*41
final_list = ls + lsac + lsas+ lsfe + lslb + lsar
df[37]= final_list
df = df.drop([0, 41, 82, 123, 164, 205])
df = df.reset_index(drop=True)
i = 0
df[38]=['xyz']*240
while(i<=235):
	if i!=0:
		df.drop([i+1], inplace=True)
	df[38][(i+2)] = df[0][i]
	df[38][(i+3)] = df[0][i]
	df[38][(i+4)] = df[0][i]
	df.drop([i], inplace=True)
	i = i +5
df = df.reset_index(drop=True)
new_df = pd.DataFrame(columns=['patient_id_junk', 'type', 'experiment', 'intervention', 'value'])
for i in range(1,37):
	for j in range(1,145):
		new_df = new_df.append({'patient_id_junk': df[i][0], 'type': df[0][j],  'experiment':df[37][j], 'intervention': df[38][j], 'value': df[i][j]}, ignore_index=True)
split_df = pd.DataFrame(new_df.patient_id_junk.str.split(' ',1).tolist(),
                                 columns = ['patient_id','level'])
result = pd.concat([split_df, new_df], axis=1)
result.drop(axis=1, columns=['patient_id_junk'], inplace=True)
result.to_csv("processed_data.csv", index=False)
'''
pid instrumetted ls intervention value					
'''
