# MIU_Thesis_CSE
Steps:
1. Install Miniconda
2. Create a new environment from your vscode terminal using following commands
    ''' 
    conda create --name your_env_name python==3.10
    '''
3. Activate your environment
4. Install packages by executing following command (Beware of the path 
    (/MIU_Thesis_CSE/...))
    ''' 
    pip install -r requirements.txt
    ''' 
    & 
    '''
    pip install -U langchain-community 
    '''
     & 
    '''
    pip install python-docx openai chromadb tiktoken
    '''

5. After installation all packages please create .env file in MIU_Thesis_CSE folder
6. Put API Key in .env file
7. Execute/run create_database.py
8. Finally in terminal execute following command(Beware of the path 
    (/MIU_Thesis_CSE/...)):
    ''' 
    streamlit run query_data.py
    '''