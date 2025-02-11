from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

#Web Search_Agent
web_search_agent=Agent(
    name="Web_Agent",
    role="find the information of stock on internet",
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
    role="Analyze financial data for the stock",  
    tools=[  
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,  
                      company_news=True),  
    ],  
    instructions=["Use bullet points"],  
    show_tool_calls=True,  
    markdown=True,  
)
multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    instructions=["Always include sources", "Use bullet points"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)