'''
	Stevens Institute of Technology
'''

# import requests package
import requests                   
# import BeautifulSoup from package bs4 (i.e. beautifulsoup4)
from bs4 import BeautifulSoup    
import re

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import numpy as np
import os
import time

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import distance

def web_scrape(address):
    # store user_name 
    nicks=[]
    # store recommendation information
    titles=[]
    # store hours the player played
    hours=[]
    # store review information
    comments=[]
    # store the number of people who thinks that this review is helpful
    helpfuls=[]
    # store the number of people who thinks that this review is funny
    funnys=[]
    # store the date the user commented
    dates=[]
    # store the number of games the user has
    products=[]
    # scrape 10 reviews per time
    # therefore we scraped 10000 reviews
    for i in range(1, 1001):
        # simulate the network requests
        url =  address + '/homecontent/?userreviewsoffset=' + str(10 * (i - 1)) + '&p=' + str(i) + '&workshopitemspage=' + str(i) + '&readytouseitemspage=' + str(i) + '&mtxitemspage=' + str(i) + '&itemspage=' + str(i) + '&screenshotspage=' + str(i) + '&videospage=' + str(i) + '&artpage=' + str(i) + '&allguidepage=' + str(i) + '&webguidepage=' + str(i) + '&integratedguidepage=' + str(i) + '&discussionspage=' + str(i) + '&numperpage=10&browsefilter=toprated&browsefilter=toprated&appid=435150&appHubSubSection=10&l=senglish&filterLanguage=default&searchText=&forceanon=1'
        html = requests.get(url).text.replace('<br>',' ')
        soup = BeautifulSoup(html, 'html.parser') 
        # scrape the information which we focus on
        reviews = soup.find_all('div', {'class': 'apphub_Card'})    
        for review in reviews:
            nick = review.find('div', {'class': 'apphub_CardContentAuthorName'})
            nicks.append(nick.text)
            title = review.find('div', {'class': 'title'}).text
            titles.append(title)
            hour = review.find('div', {'class': 'hours'}).text.split(' ')[0]
            hours.append(hour)
            product = review.find('div', {'class': 'apphub_CardContentMoreLink ellipsis'}).text.split()
            # this content may be null. when it is null, we think it equals to zero
            if len(product)==4:
                products.append(product[0])
            else:
                products.append('0')
            #link = nick.find('a').attrs['href']
            comment = review.find('div', {'class': 'apphub_CardTextContent'}).text
            temp=comment.split('\n')
            # there will be unwanted information. so we skip them.
            if len(temp)==3:
                comments.append(temp[2].strip('\t'))
            else:
                comments.append(temp[3].strip('\t'))
            # delete string "Posted: " since it is unused
            date = re.sub(r"Posted:"," ",comment.split('\n')[1].strip('\t')).strip()
            dates.append(date)
            helpful = review.find('div', {'class': 'found_helpful'}).text.split()[0]
            helpfuls.append(helpful)
            #helpful1 = review.find('div', {'class': 'found_helpful'}).text.split()[6]
            #funny = re.findall(r"\d+",review.find('div', {'class': 'found_helpful'}).text.split()[5])
            funny = review.find('div', {'class': 'found_helpful'}).text.split()
            # this content may be null. when it is null, we think it equals to zero
            if len(funny)==12:
                funnys.append(funny[6])
            else:
                funnys.append('0')
                
    # generate dataframe to store web information
    df = pd.DataFrame()
    df["names"] = nicks
    df["products#"] = products
    df["marked as helpful"] = helpfuls
    df["marked as funny"] = funnys
    df["post_date"] = dates
    df["Recommend?"] = titles
    df["times on record"] = hours
    df["review"] = comments
    
    # generate csv file
    # df.to_csv('game_data_negative.csv',index=False)
    return df

def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ

    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB

    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN

    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN


def tokenize(data):
    
    # Fomart the date       
    init_dates = []
    date_lst = data['post_date'].tolist()
    for i in range(len(data)):
        temp=date_lst[i].split(',')
        # when the review was posted in 2018, the system doesn't show "2018". We add it.
        if len(temp)==1:
            temp.append(' 2018')
        if len(temp)==2:
            date_temp=''.join(temp)
            conv=time.strptime(date_temp,"%B %d %Y")
            date_temp2=time.strftime("%Y/%m/%d",conv)
            init_dates.append(date_temp2)
    data["post_date"]=init_dates
    
    # Lemmatization: determining the lemma for a given word
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Regular expression pattern
    pattern = r'\w[\w\'-]*\w'
    
    init_reviews = []
    init_helpfuls = []
    stop_words = stopwords.words('english')
    stop_words.append("game")
    
    review = data["review"].values.tolist()
    helpfuls = data["marked as helpful"].values.tolist()
    
    for doc in review:
        doc = doc.lower()
        tokens = nltk.regexp_tokenize(doc, pattern)
        tagged_tokens= nltk.pos_tag(tokens)
        lemmatized_words=[wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag))               for (word, tag) in tagged_tokens               if word not in stop_words and word not in string.punctuation]
        temp_str = ""
        for item in lemmatized_words:
            temp_str = temp_str + " " + item
        init_reviews.append(temp_str[1:])
    
    # convert "No" to "0"
    for line in helpfuls:
        if line=="No":
            init_helpfuls.append("0")
        else:
            init_helpfuls.append(line)
    
    
        
    # Generate csv file to save data without unuseful information.
    df = pd.DataFrame()
    df["user_name"] = data["names"].values.tolist()
    df["user_product"] = data["products#"].values.tolist()
    df["helpful"] = init_helpfuls
    df["funny"] = data["marked as funny"].values.tolist()
    df["post_date"] = data["post_date"].values.tolist()
    df["recommend_or_not"] = data["Recommend?"].values.tolist()
    df["game_time"] = data["times on record"].values.tolist()
    df["review"] = init_reviews
    # obtain the row index when contents of reviews are empty
    indx = df[df.review==''].index.tolist()
    # delete the corresponding datasets
    df1=df.drop(df.index[indx])
    # delete the duplicate datasets
    df1.drop_duplicates(subset ="user_name", 
                     keep = False, inplace = True)
    #sort df by date value
    df2=df1.sort_values(by='post_date')
    # csv
    df2.to_csv('C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/Data_main/tokened_normed_review_test.csv',index=False)
    # this method generate a dataframe with tokenized data sets
    return df2

def merge_csv():
    Folder_Path = r'C:\Users\yongk\Documents\PythonLearning\Steam Data Analysis\Data_main'          
    SaveFile_Path =  r'C:\Users\yongk\Documents\PythonLearning\Steam Data Analysis\Data_main'       
    SaveFile_Name = r'game_data_all.csv'              
 
    os.chdir(Folder_Path)
    # save file names into a list
    file_list = os.listdir()
 
    # read the first csv including headers
    df = pd.read_csv(Folder_Path +'\\'+ file_list[0])   #default utf-8
 
    # write the first csv to second one
    df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False)
 
    # loop through all files
    for i in range(1,len(file_list)):
        df = pd.read_csv(Folder_Path + '\\'+ file_list[i])
        df.to_csv(SaveFile_Path+'\\'+ SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')


def getdata():
    # obtain data including positive and negative reviews
    data_positive = web_scrape(address_positive)
    data_negative = web_scrape(address_negative)
    # concat two dataframes
    data = pd.concat([data_positive,data_negative],ignore_index=True)
    # generate csv files to store two dataframes
    data_positive.to_csv('C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/Data_main/game_data_positive.csv',index=False)
    data_negative.to_csv('C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/Data_main/game_data_negative.csv',index=False)
    # merge multiple csv
    merge_csv()# game_data_all.csv
    return data

def naive_approach(token_path,posi_word_path,nega_word_path):
    '''
    This function take a csv file with tokenized review column as input
    and return a dataframe with additional columns 'positive freq' and 'negative freq'
    Args:
        token_path: file path of 'tokened_normed_review_v2.csv'
        posi_word_path: file path of 'positive-words.txt'
        nega_word_path: file path of 'negative-words.txt'
    Return:
        data: dataframe created from the csv file with two new columns 'positive freq' and 'negative freq'
    '''

    data = pd.read_csv(token_path,header = 0)
    with open(posi_word_path, 'r') as f1:
        positive_words = [line.strip() for line in f1]
    with open(nega_word_path, 'r') as f2:
        negative_words = [line.strip() for line in f2]
        
    reviews = list(data['review'])
    positive_tokens = []
    negative_tokens = []
    for doc in reviews:
        doc_positive_tokens = [token for token in doc.split()
                           if token in positive_words]
        positive_tokens.append(len(doc_positive_tokens)/len(doc))

        doc_negative_tokens = [token for token in doc.split()
                          if token in negative_words]
        negative_tokens.append(len(doc_negative_tokens)/len(doc))
        
    data['positive freq'] = positive_tokens
    data['negative freq'] = negative_tokens
    data['attitude']=data['positive freq']-data['negative freq']
    return data


# This function is to tokenalize normal review before we do TF-IDF #
def get_doc_tokens(doc):
    stop_words = stopwords.words('english')
    stop_words.append("game")
    tokens=[token.strip()             for token in nltk.word_tokenize(doc) if token.strip() not in stop_words and               token.strip() not in string.punctuation]
    # create token count dictionary
    token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count


# This function is to get TF-IDF matrix #
def get_tf_idf(reviews):
    
    docs_tokens={idx:get_doc_tokens(doc) for idx,doc in enumerate(reviews)}

    # since we have a small corpus, we can use dataframe to get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
    # convert dtm to numpy arrays
    tf=dtm.values

    # sum the value of each row
    doc_len=tf.sum(axis=1)

    # divide dtm matrix by the doc length matrix
    tf=np.divide(tf, doc_len[:,None])

    # get document freqent
    df=np.where(tf>0,1,0)

    # get idf
    smoothed_idf=np.log(np.divide(len(reviews)+1, np.sum(df, axis=0)+1))+1

    # get tf-idf
    smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf

# This function is to get the most similarity review given a review_id #
def find_similar_doc(doc_id, smoothed_tf_idf):
    similarity=1-distance.squareform(distance.pdist(smoothed_tf_idf, 'cosine'))
    
    # find top doc similar to first one
    best_matching_doc_id = np.argsort(similarity)[:,::-1][doc_id,0:2][1]
    similarity = similarity[doc_id,best_matching_doc_id]  
    return best_matching_doc_id, similarity


# ## 4. Test


if __name__ == "__main__":
    
# Test "Scrape Data from Steam Platform"
    # Data from "Divinity: Original Sin 2"
    address_positive = 'https://steamcommunity.com/app/435150/positivereviews'
    # https://steamcommunity.com/app/435150/
    address_negative = 'http://steamcommunity.com/app/435150/negativereviews'
    data = getdata() # get raw datasets
    
    # output
    # there are some problems about the output (posive, negative)
    #print("Quantities of positive reviews")
    #print(len(web_scrape(address_positive)))
    #print("Quantities of negative reviews")
    #print(len(web_scrape(address_negative)))
    print("Quantities of all reviews")
    print(len(data))
    print("\n")
    
# Test "Data Tokenization and normalization"
    tokenized_data = tokenize(data)
    # output
    print("Quantities of reviews after tokenization (remove unuseful infomation)")
    print(len(tokenized_data))
    print('\n')
# Test "Data Analysis"
    # Sentiment Analysis
    
    # TF-IDF Analysis
    reviews = tokenized_data["review"].values.tolist()
    tf_idf = get_tf_idf(reviews)
    print("Smoothed TF-IDF Matrix")
    print(tf_idf)
    
    print(find_similar_doc(1,tf_idf))
	
	# scatter plot
	token_path = r'C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/tokened_normed_review_v2.csv'
	posi_word_path = r'C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/positive-words.txt'
	nega_word_path = r'C:/Users/yongk/Documents/PythonLearning/Steam Data Analysis/negative-words.txt'
	data = naive_approach(token_path,posi_word_path,nega_word_path)
	data.user_product=data.user_product.apply(lambda x :x.replace(',','')).astype(int)
	data.helpful=data.helpful.apply(lambda x :x.replace(',','')).astype(int)
	data.funny=data.funny.apply(lambda x :x.replace(',','')).astype(int)
	data.game_time=data.game_time.apply(lambda x :x.replace(',','')).astype(float)
	data['review_len']=data.review.apply(lambda x: len(x))
	data.head()
	get_ipython().run_line_magic('matplotlib', 'inline')
	sns.set_style("whitegrid");
	sns.pairplot(data=data, 
				x_vars=['user_product','helpful','funny','game_time','review_len','attitude'], 
				y_vars=['user_product','helpful','funny','game_time','review_len','attitude'], 
				hue='recommend_or_not')

	
	
	

# - this is not a function, but you can run it independently to test the performance of this method.

# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions
# import time

# executable_path = 'F:\geckodriver'

# driver = webdriver.Firefox(executable_path=executable_path)

# driver.get('https://steamcommunity.com/app/435150/reviews/?p=1&browsefilter=toprated')

# src_updated = driver.page_source
# src = ""

# for i in range(0,190):
#     if src != src_updated:
#         # save page source (i.e. html document) before page-down
#         src = src_updated
#         # execute javascript to scroll to the bottom of the window
#         # you can also use page-down
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         # sleep to allow content loaded
#         time.sleep(1)
#         # save page source after page-down
#         src_updated = driver.page_source


# # What can we get about the reviews from Steam?

# # The review content, post date, how many people think this review helpful or funny

# review_content=[]
# usefuls=[]

# for i in range(1,191):
#     # get all Q&A list using XPATH locator
#     lists=driver.find_elements_by_xpath("//div[@id='page%d']/div"%i)
# #     print("//div[@id='page%d']/div"%i)
# #     print("page%d pairs: "%i,len(lists))
    
#     for idx,item in enumerate(lists):    
#     # each Q&A pair has an unique ID
#         div_id=item.get_attribute("id")
#         content_css="div#"+div_id+" "+"div.apphub_UserReviewCardContent div.apphub_CardTextContent"
#         useful_css="div#"+div_id+" "+"div.apphub_UserReviewCardContent div.found_helpful"
        
#         review=driver.find_element_by_css_selector(content_css)
#         useful=driver.find_element_by_css_selector(useful_css)
#         review_content.append(review.text)
#         usefuls.append(useful.text)

# print("Total reviews scraped: ", len(review_content), len(usefuls))
# # print(review_content[0])
# # print(usefuls[0])

# import pandas as pd
# import nltk
# import re

# reviews =[]
# dates = []

# # print(type(data[0]),str(data[0]))

# for line in review_content:
#     sentence = line.split("\n")
#     date = re.sub(r"Posted:"," ",sentence[0]).strip()
#     review = re.sub(r"Posted:.*"," ",line)
#     review = re.sub(r"\s+"," ",review).strip() 
#     reviews.append(review)
#     dates.append(date)
    
# # print("date:",dates[0])
# # print("content:",reviews[0])



# df = pd.DataFrame()
# df["review"] = reviews
# df["post_date"] = dates
# df["useful"] = usefuls
# df.to_csv('game_review.csv',index=True)
