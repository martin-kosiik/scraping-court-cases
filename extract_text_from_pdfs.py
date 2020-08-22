import os
import re
import pdfplumber
import pandas as pd
# set your current working directory

working_directory = "C:/Users/marti/OneDrive/Plocha/RA_work/scraping_court_cases"

os.chdir(working_directory)

#print(type(os.listdir('pdf_files/2013')[1]))

print(re.match(' \(1\)\.pdf$', 'А56-55523-2013__20140221 (1).pdf'))
print(re.match('dd', 'А56-55523-2013__20140221 (1).pdf'))


print('А56-55523-2013__20140221 (1).pdf'[-8:])
#%% delete the duplicate pdfs

folders_of_pdfs = os.listdir('pdf_files')
for folder in folders_of_pdfs:
    all_pdf_files_in_folder = os.listdir(os.path.join(working_directory, 
                                                      "pdf_files", folder))
    
    for pdf_file in all_pdf_files_in_folder:
        if pdf_file[-8:] ==  ' (1).pdf':
            #os.remove(os.path.join(working_directory, "pdf_files", folder, pdf_file))
            print(pdf_file)
        #print(pdf_file)
    #print(os.path.join(working_directory, folder))
            


# %%
extracted_text_list = []
case_id_list = []
year_list = []

folders_of_pdfs = os.listdir('pdf_files')

for folder in folders_of_pdfs:
    all_pdf_files_in_folder = os.listdir(os.path.join(working_directory, 
                                                      "pdf_files", folder))
    for pdf_file in all_pdf_files_in_folder:
                    
        with pdfplumber.open(os.path.join("pdf_files", folder, pdf_file)) as pdf:
            all_text = ''
            
            for pdf_page in pdf.pages:
                single_page_text = pdf_page.extract_text()
                all_text = all_text + 'NEWPAGE \n' + single_page_text
        
        extracted_text_list.append(all_text)
        case_id_list.append(pdf_file[:-4])   # remove the .pdf characters from the str.
        year_list.append(int(folder))
    print('Folder ' + folder + ' finished.')
        

#%%
arbitrage_rulings_df = pd.DataFrame ({'case_id': case_id_list,
                                    'year': year_list,
                                    'text': extracted_text_list})

arbitrage_rulings_df.to_csv('arbitrage_rulings.csv', index=False, encoding="UTF-8")      
# %%

arbitrage_rulings_df = pd.read_csv("arbitrage_rulings.csv", encoding="UTF-8")
