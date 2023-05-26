# jobfinder
Find jobs given a linkedin profile and write a cover letter for them.

This program uses LangChain, OpenAI, SERP API, and Apify to run the functions. 

To make this system run, enter your OpenAI API key, Apify API key, and SERP API key on the side bar. Then enter a LinkedIn profile URL and let the system do the rest. 

When clicking a checkbox for a recommended position, this will trigger the SERP api to find jobs on Google for that position. Opening each expander, you can read about the qualifications, benefits, and requirements for each job. Then, if you click on "Make Me a Cover Letter", it will load a Cover Letter template that you can review and submit for that position based on the experience of the LinkedIn Profile. 
