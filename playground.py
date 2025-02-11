from phi.agent import Agent
from phi.model.groq import Groq

from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground,serve_playground_app
load_dotenv()

phi.pi=os.getenv("PHI_API_KEY")

web_search_agent=Agent(
    name="Web_Agent",
    role="find the information of stock ",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],  
    instructions=["Always include sources"],  
    show_tools_calls=True,  
    markdown=True, 
)
#Finance Agent
finance_agent = Agent(  
    name="Finance_Agent",  
    model=Groq(id="deepseek-r1-distill-llama-70b"),  
    tools=[  
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,  
                      company_news=True),  
    ],  
    instructions=["Use bullet points"],  
    show_tool_calls=True,  
    markdown=True,  
)

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)