from bs4 import BeautifulSoup
import requests
import os
import urllib.request


os.chdir("C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases")

with open("first_page.html", encoding='utf8') as html_file:
    soup = BeautifulSoup(html_file, 'html5lib')


print(soup.title)


# %%
# <a class="b-a-blue js-popupDocumentShow" target="_blank" href="https://ras.arbitr.ru/Kad/PdfDocument/e0edcd7e-ae66-4aed-90d1-6ef3cfd3a6f0/4c4d0cfb-620f-48b4-872c-92b1d9116e68/А32-1381-2020__20200729.pdf">       
# #b-cases > li:nth-child(9) > div.doc > div.doc-text > a
#print(soup.select("#b-cases > li:nth-child(9) > div.doc > div.doc-text > a"))                                                                                                         
#print(soup.select("#b-cases > li > div.doc > div.doc-text > a"))                                                                                                         

all_links = soup.select("#b-cases > li > div.doc > div.doc-text > a")

all_links_to_pdfs = []

for link in all_links:
    all_links_to_pdfs.append(link['href'])

print(all_links[5]['href'])
#print(soup)

#%%
print(all_links_to_pdfs[5])

filename = os.path.join(os.path.join(os.getcwd(), "pdf_files"),
                           "page_1_id_" + str(5 +8) + ".html" )

url_link = 'https://ras.arbitr.ru/Document/Pdf/fb56856f-fd51-4ed2-98c8-f964718d3071/6c1279c9-c2fc-4a53-a5e8-722bddf4154d/%D0%9037-817-2020__20200814.pdf'

with open(filename, 'wb') as f:
        f.write(requests.get(url_link, allow_redirects=True).content)
  
#%%

# https://ras.arbitr.ru/Ras/HtmlDocument/fa57a4e0-00a3-4e9f-a065-3b52ab8eea67
        
# %%
filename = os.path.join(os.path.join(os.getcwd(), "pdf_files"),
                          "page_1_id_" + str(8 + 1) + ".pdf" )
urllib.request.urlretrieve(all_links_to_pdfs[5], filename)

# %%
import wget

wget.download(all_links_to_pdfs[5], filename)

     
# %%
url_name = 'https://scholar.princeton.edu/sites/default/files/lwantche/files/the_slave_trade_and_the_origins_of_mistrust_in_africa_use_this_one.pdf'
print(urllib.request.urlparse(all_links_to_pdfs[5]))
   
with open(filename, 'wb') as f:
        f.write(requests.get(url_name, allow_redirects=True).content)
#%%
print(len(all_links_to_pdfs))
print(len(all_links))


for i in range(len(all_links_to_pdfs)):
    #Name the pdf files using the last portion of each link which are unique in this case
    filename = os.path.join(os.path.join(os.getcwd(), "pdf_files"),
                           "page_1_id_" + str(i + 1) )
    print(filename)
    #with open(filename, 'wb') as f:
    #    f.write(requests.get(urljoin(url,link['href'])).content)
