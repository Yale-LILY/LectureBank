
import csv,os,shutil,wget,httplib2
import subprocess as sp
from pathlib import Path

base_path='lecturebank_files/'
data_path='lecturebank.tsv'

file_format=['pdf','pptx','p']

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
        resource_id= row[0]
        resource_format=row[2].split('.')[-1]

        if resource_format in file_format:
            if resource_format == 'p':
                resource_format = 'pdf'
            download_name = resource_id+'.'+resource_format

            resource_dict[download_name] = resource_url
            count += 1



print('In total..',count)


#check base path
if os.path.exists(base_path):
    shutil.rmtree(base_path)

os.makedirs(base_path)

print ('Base path dir made..',base_path)

print ('Now start downloading ....')
count_fail = 0
count_success = 0
failed_list =[]
for key,item in resource_dict.items():

    out_file_path = os.path.join(base_path,key)

    download_url = item

    try:

        wget.download(download_url, out_file_path)
        count_success += 1

        print ('Downloading {} out of {}'.format(count_success,len(resource_dict)))
    except:
        failed_list.append(out_file_path)
        count_fail += 1
        print ('URL invalid')

print ('\n\nFinished downloading ...',count_success)
print ('URL breaks on ...', count_fail)

for x in failed_list:
    print (x)
