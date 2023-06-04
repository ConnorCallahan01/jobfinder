import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.document_loaders import ApifyDatasetLoader
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from serpapi import GoogleSearch
from langchain.chains import RetrievalQA
import streamlit.components.v1 as components
from streamlit.components.v1 import html


with st.sidebar:
    st.subheader("Enter your API credentials: ")
    open_api_key = st.text_input("Enter your Open API Key")
    apify_api_key = st.text_input("Enter your Apify API Key")
    serp_api_key = st.text_input("Enter your SERPAPI Key")
    st.write("Support my projects, buy me a coffee! [link](https://bmc.link/pcallahan)")
    components.html("""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                            <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="large" data-theme="light" data-type="HORIZONTAL" data-vanity="connor-callahan" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://www.linkedin.com/in/connor-callahan?trk=profile-badge"></a></div>
                """, height=280)
os.environ["OPENAI_API_KEY"] = open_api_key 
os.environ["APIFY_API_TOKEN"] = apify_api_key 

@st.cache_resource
def load_patrick(url): 
    
    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": url}]},
        dataset_mapping_function=lambda item: Document(
            page_content=item["text"] or "", metadata={"source": item["url"]}
        ),
    )
    # index = VectorstoreIndexCreator().from_loaders([loader])
    data=loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    st.write("1: LinkedIn Data Loaded")
    
    texts = text_splitter.split_documents(data)
    st.write("2: Data has been vectored.")
    
    
    return texts
@st.cache_data
def question(_texts):
    prompt_template = """You are a job search assistant for the person. You will be given their LinkedIn profile and they
    will ask a question about which jobs they will be qualified for. Given their LinkedIn profile, answer the question
    to best of your abilities that will help them with their job search. 

    

    Task: This person is looking for a new job position. Return a list of 5 different job titles that this person doesn't have but could consider and would succeed at given they are searching for another job.
    Output Format: Python List. Example of format: '[job_1, job_2, job_3, job_4, job_4]"""
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(_texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(_texts))]).as_retriever(search_kwargs={"k": 2})
    docs = docsearch.get_relevant_documents(prompt_template)
    
    # PROMPT = PromptTemplate(
    #     template=prompt_template
    # )


    st.write("3: Searching for recommended job titles...")
    chain = load_qa_chain(OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0, model_name="text-davinci-003"), chain_type="map_reduce")
    outcome = chain({"input_documents": docs, "question": prompt_template}, return_only_outputs=True)
    return outcome['output_text']


def coverLetter(documents, name, role, description):
    embeddings = OpenAIEmbeddings()
    texts = documents
    docsearch = Chroma.from_documents(texts, embeddings)
    prompt_template = f"Your job is to take in the LinkedIn profile provided and to generate a cover letter for that person based on the job title, company name, and role description. The cover letter should be polite, professional, and include the person's experience that would make them a great role for the position. Company name: {name}, Job Title: {role}, Job Description: {description}"
    qa = RetrievalQA.from_chain_type(llm=OpenAI(max_tokens=550), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))         
    result = qa(prompt_template)
    st.write(result["result"])


st.header("Job Search Assistant")
st.caption("""Enter in a LinkedIn URL. It'll give you a list of 5 options for potential job titles. """)
st.caption("""When clicking on the check for each title, it will give 10 expanders with job role and company name. Clicking on each expander reveals requirements, qualifications, and benefits for that position. """)
st.caption("""As the bottom of each job expander, you'll see a button that says 'Make me a Cover Letter'. Upon clicking this button, it will make a cover letter for that job position with mentions to the candidate's experience based from the LinkedIn profile.""")
st.caption("""NOTE: these cover letters are meant to be a template for what the user could use. It should be edited and not just copy and pasted.""")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

linkedin_url = st.text_input(
    "Enter LinkedIn URL 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder="LinkedIn URL",
)
if linkedin_url:
    st.write("You're LinkedIn URL: ", linkedin_url)
    docsearch = load_patrick(linkedin_url)
    
    # query = st.text_input(
    #     "Enter a Question 👇",
    #     label_visibility=st.session_state.visibility,
    #     disabled=st.session_state.disabled,
    #     placeholder="Who does the person work for?",
    # )
    # if query: 
    output = question(docsearch)
    st.write(output)
    res = output.strip('][').split(', ')
    st.write("These are the recommended job positions. Check the ones you'd like to search for job positions for yourself.") 
    check1 = st.checkbox(res[0])
    check2 = st.checkbox(res[1])
    check3 = st.checkbox(res[2])
    check4 = st.checkbox(res[3])
    check5 = st.checkbox(res[4])

    search_list = []
    if check1:
        search_list.append(res[0])
    if check2:
        search_list.append(res[1])
    if check3:
        search_list.append(res[2])
    if check4:
        search_list.append(res[3])
    if check5:
        search_list.append(res[4])
    
    for i in search_list:
        #serp_api parameters
        st.header(f"Results for: {i}")
        params = {
        "engine": "google_jobs",
        "google_domain": "google.com",
        "q": i,
        "api_key": serp_api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        count = 0
        for i in results["jobs_results"]:
            count+=1
            title = i["title"]
            company = i["company_name"]
            description = i["job_highlights"]
            with st.expander(":red[Role:] "+i["title"]+", :red[Company Name:] "+i["company_name"]):
                st.write(i["job_highlights"])
                if st.button('Make me a cover letter!', key=count):
                    coverLetter(docsearch, title, company, description)


               
