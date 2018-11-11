
import csv,os,shutil
import subprocess as sp
from pathlib import Path

URL_list = []

base_path='data_lecturebank/'
data_path='lecturebank.tsv'



count = 0
data = []

# organize the urls by the bype of the resources: ir, ai, nlp, etc
resource_dict = {}

with open(data_path, "r") as resources:
    reader = csv.reader(resources,delimiter='\t')
    next(reader)
    for row in reader:
        resource_type = row[-2]
        resource_url = row[2]
        resource_school = row[-1]
        if resource_type in resource_dict.keys():
            resource_dict[resource_type] += [resource_url+'|'+resource_school]
        else:
            resource_dict[resource_type] = [resource_url+'|'+resource_school]

        count +=1

print ('A list of urls found:')
for key,item in resource_dict.items():
    print (key, len(item))

print('In total..',count)


#check base path
if os.path.exists(base_path):
    shutil.rmtree(base_path)
else:
    os.makedirs(base_path)

print ('Base path dir made..',base_path)

print ('Now start downloading ....')

count_fail = 0
for key,item in resource_dict.items():

    out_path = os.path.join(base_path,key)
    os.makedirs(out_path)
    print ('Now downloading...',key)

    url_list = item
    for id,url in enumerate(url_list):

        name = str(id)+"_"+url.split('|')[-1]+'_'+ Path(url.split('|')[0]).name
        url = url.split('|')[0]

        command = ["wget", "-O", os.path.join(out_path,name), url, "--timeout=30", "--tries=3"]
        line = sp.call(command)

print ('Finish!')

