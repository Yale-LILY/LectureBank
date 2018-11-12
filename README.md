
# LectureBank: a corpus for NLP Education and Prerequisite Chain Learning

This is the github page for our paper *What Should I Learn First: Introducing LectureBank for NLP Education and Prerequisite Chain Learning* in the proceedings AAAI 2019. 
![An example of prerequisite relations from lecture slides depicted as a directed graph](https://github.com/Yale-LILY/LectureBank/blob/master/imgs/slide_diagram_final.png =80x40)

The list of descriptions can be found in the following:

### LectureBank Dataset
LectureBank Dataset is a manually-collected dataset of lecture slides. We collected 1352 online lecture files from 60 courses covering 5 different domains,  including Natural Language Processing (nlp), Machine Learning (ml), Artificial Intelligence (ai), Deep Learning (dl) and Information Retrieval (ir).  In addition, we release the corresponding annotations for each slide file to the taxonomy described below.  We also provide an additional vocabulary list of size 1221 extracted from the corpus.  

### lecturebank.tsv
Each line identifies a lecture file. Format:

`(ID, Title, URL, Topic_ID, Year, Author, Domain, Venue)`


- ID: Id of each line.
- Title: File tile.
- URL: Online URL.
- Topic_ID: Classified taxonomy Topic ID, referring topics from taxonomy.tsv.
- Year: Year of the course.
- Author: The author name(s).
- Domain: The domain (nlp, ir, dl, ml, ai).
- Venue: Name of the university, or 'GitHub'.

### download_all.py
The scripts of downloading the resources from the urls of `lecturebank.tsv`. After running the scripts, all the resources will be downloaded into `data_lecturebank/` folder, organized by the `Domain` (for example, nlp, ir). 

### taxonomy.tsv
Contains taxonomy topics and corresponding IDs referred by `lecturebank.tsv`.


### 208topics.csv
Contains the 208 topics which we annotated, format:

`(ID, Topic, Wiki_Page_URL)`

### prerequisite_annotation.csv
Contains the prerequisite chain annotation for each possible pair from the 208 topics. Format:

`(Source_Topic_ID, Target_Topic_ID, If_prerequisite)`


### vocabulary.txt
Contains 1221 vocabulary terms combined from taxonomy, 208 topics and terms extracted from LectureBank.










