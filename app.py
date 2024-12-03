import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain and OpenAI Imports
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish

# Load environment variables
load_dotenv()

class CompanyResearchTool(BaseTool):
    name: str = Field(default="company_research_tool")
    description: str = Field(default="Conducts comprehensive research on a specific company and its industry")

    def _run(self, company_name: str) -> str:
        """
        Perform in-depth research on the company
        """
        try:
            # OpenAI API call for detailed company research
            llm = ChatOpenAI(
                model='gpt-4-turbo', 
                temperature=0.7, 
                max_tokens=4000
            )
            
            # Detailed research prompt
            research_prompt = PromptTemplate(
                input_variables=['company_name'],
                template="""
                Conduct a comprehensive research report on {company_name} covering:
                1. Company Overview
                   - Industry sector
                   - Primary business activities
                   - Market positioning
                
                2. Technological Landscape
                   - Current technology stack
                   - Digital transformation initiatives
                   - Technology challenges
                
                3. Potential AI/ML Transformation Areas
                   - Operational inefficiencies
                   - Customer experience gaps
                   - Innovation opportunities
                
                Provide a detailed, structured analysis that highlights strategic AI/ML implementation possibilities.
                """
            )
            
            # Create research chain
            research_chain = LLMChain(
                llm=llm, 
                prompt=research_prompt
            )
            
            # Execute research
            research_result = research_chain.run(company_name=company_name)
            return research_result
        
        except Exception as e:
            return f"Research error: {str(e)}"
    
    def _arun(self, company_name: str) -> str:
        """Async run method"""
        raise NotImplementedError("Async method not supported")

class UseCaseGenerationTool(BaseTool):
    name: str = Field(default="use_case_generation_tool")
    description: str = Field(default="Generates AI and GenAI use cases based on company research")

    def _run(self, company_research: str) -> str:
        """
        Generate AI use cases based on company research
        """
        try:
            llm = ChatOpenAI(
                model='gpt-4-turbo', 
                temperature=0.7, 
                max_tokens=4000
            )
            
            # Use case generation prompt
            use_case_prompt = PromptTemplate(
                input_variables=['company_research'],
                template="""
                Based on the following company research, generate 20 comprehensive AI and GenAI use cases:
                {company_research}
                
                For each use case, provide:
                1. Use Case Title
                2. Objective/Use Case Description
                3. AI/ML Application Strategy
                4. Cross-Functional Benefits
                
                Focus on:
                - Operational efficiency
                - Customer experience enhancement
                - Innovation and digital transformation
                - Cost reduction
                - Risk management
                """
            )
            
            # Create use case generation chain
            use_case_chain = LLMChain(
                llm=llm, 
                prompt=use_case_prompt
            )
            
            # Generate use cases
            use_cases = use_case_chain.run(company_research=company_research)
            return use_cases
        
        except Exception as e:
            return f"Use case generation error: {str(e)}"
    
    def _arun(self, company_research: str) -> str:
        """Async run method"""
        raise NotImplementedError("Async method not supported")

class ResourceAssetCollectionTool(BaseTool):
    name: str = Field(default="resource_asset_collection_tool")
    description: str = Field(default="Collects relevant datasets and resources for proposed use cases")

    def _run(self, use_cases: str) -> str:
        """
        Collect resources for AI use cases
        """
        try:
            llm = ChatOpenAI(
                model='gpt-4-turbo', 
                temperature=0.7, 
                max_tokens=4000
            )
            
            # Resource collection prompt
            resource_prompt = PromptTemplate(
                input_variables=['use_cases'],
                template="""
                For the following AI use cases, find and list:
                1. Relevant datasets from Kaggle, HuggingFace, GitHub
                2. Open-source tools and libraries
                3. Potential implementation resources
                
                Use Cases:
                {use_cases}
                
                For each use case, provide:
                - Dataset links
                - Open-source tool recommendations
                - Implementation guide references
                """
            )
            
            # Create resource collection chain
            resource_chain = LLMChain(
                llm=llm, 
                prompt=resource_prompt
            )
            
            # Collect resources
            resources = resource_chain.run(use_cases=use_cases)
            return resources
        
        except Exception as e:
            return f"Resource collection error: {str(e)}"
    
    def _arun(self, use_cases: str) -> str:
        """Async run method"""
        raise NotImplementedError("Async method not supported")

def generate_comprehensive_report(company_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive AI use case report
    """
    # Initialize tools
    company_research_tool = CompanyResearchTool()
    use_case_generation_tool = UseCaseGenerationTool()
    resource_asset_tool = ResourceAssetCollectionTool()
    
    # Conduct company research
    company_research = company_research_tool.run(company_name)
    
    # Generate use cases
    use_cases = use_case_generation_tool.run(company_research)
    
    # Collect resources
    resources = resource_asset_tool.run(use_cases)
    
    return {
        "company_name": company_name,
        "research": company_research,
        "use_cases": use_cases,
        "resources": resources
    }

def save_report_to_file(report: Dict[str, Any]):
    """
    Save generated report to a markdown file
    """
    filename = f"{report['company_name'].replace(' ', '_')}_ai_use_cases.md"
    with open(filename, 'w') as f:
        f.write(f"# AI Use Cases for {report['company_name']}\n\n")
        f.write("## Company Research\n")
        f.write(report['research'])
        f.write("\n\n## AI Use Cases\n")
        f.write(report['use_cases'])
        f.write("\n\n## Resources and Implementation Guides\n")
        f.write(report['resources'])
    return filename

def main():
    st.title("🚀 AI Use Case Generation Platform")
    
    # Company input
    company_name = st.text_input("Enter Company Name", placeholder="e.g., Tesla, Microsoft")
    
    if st.button("Generate AI Use Case Report"):
        if company_name:
            with st.spinner("Generating Comprehensive AI Use Case Report..."):
                try:
                    # Generate report
                    report = generate_comprehensive_report(company_name)
                    
                    # Save report
                    report_path = save_report_to_file(report)
                    
                    # Display sections
                    st.subheader("Company Research")
                    st.write(report['research'])
                    
                    st.subheader("AI Use Cases")
                    st.write(report['use_cases'])
                    
                    st.subheader("Resources")
                    st.write(report['resources'])
                    
                    # Download report
                    with open(report_path, "r") as file:
                        st.download_button(
                            label="Download Full Report",
                            data=file,
                            file_name=report_path,
                            mime="text/markdown"
                        )
                
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
        else:
            st.warning("Please enter a company name")

if __name__ == "__main__":
    main()