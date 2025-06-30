#!/usr/bin/env python3
"""
AG2 Stock Analysis Agent System - Part 2: Basic Multi-Agent Implementation

This educational example demonstrates the core concepts of AG2 framework:
- Multiple specialized ConversableAgent instances
- Sequential chat orchestration with carryover
- LLM configuration and management
- Agent communication patterns

The system analyzes stocks using four specialized agents:
1. Data Analyst Agent - analyzes price data and calculates metrics
2. News Analyst Agent - evaluates news sentiment
3. Risk Assessment Agent - assesses market risk factors
4. Coordinator Agent - synthesizes findings into final report

Usage:
    python agent_2_simple_openai.py

Prerequisites:
    - Set up .env file with OPENAI_API_KEY
    - Install requirements: pip install "ag2[openai]" python-dotenv
"""

import os
import sys
from typing import Dict, Any
import logging

# Import AG2 framework components (installed as autogen package)
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

# Import environment configuration
from dotenv import load_dotenv

# Configure logging for educational visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAnalysisSystem:
    """
    Main orchestrator for the multi-agent stock analysis system.
    
    This class demonstrates AG2's core concepts:
    - Agent creation and configuration
    - LLM configuration management
    - Sequential chat orchestration
    - Mock data management for educational purposes
    """
    
    def __init__(self):
        """Initialize the stock analysis system with environment configuration."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Load basic configuration first
        self.verbose = os.getenv('VERBOSE_OUTPUT', 'True').lower() == 'true'
        self.max_tokens = int(os.getenv('MAX_TOKENS', '1000')) # The maximum number of tokens to generate in a response - feel free to increase this if you want more detailed responses
        self.temperature = float(os.getenv('TEMPERATURE', '0.7')) # The temperature of the response (0-1) - feel free to increase this if you want more creative responses
        
        # Validate OpenAI API key configuration
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key or self.api_key == 'your_openai_api_key_here':
            print("❌ Error: Please set OPENAI_API_KEY in your .env file")
            print("   Get your API key from: https://platform.openai.com/account/api-keys")
            sys.exit(1)
        
        if self.verbose:
            print("🔑 Using OpenAI API")
        
        # Load model configuration
        self.model = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')        
        # Configure LLM for all agents
        self.llm_config = self._setup_llm_config()
        
        # Initialize agents
        self.agents = self._create_agents()
        
        # Generate realistic mock data for educational purposes
        self.mock_data = self._generate_mock_data()
    
    def _setup_llm_config(self) -> LLMConfig:
        """
        Configure the Language Model for all agents.
        
        This demonstrates AG2's LLMConfig pattern which standardizes 
        model access across all agents in the system.
        
        Returns:
            LLMConfig: Configured LLM instance for OpenAI
        """
        if self.verbose:
            print(f"🔧 Configuring LLM: {self.model}")
        
        # OpenAI LLM configuration
        config = LLMConfig(
            model=self.model,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        return config
    

    def _generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate realistic mock data based on current NVDA market conditions.
        
        For educational purposes, we use mock data to avoid API dependencies
        while maintaining realistic values that students can relate to.
        
        Returns:
            Dict containing mock stock data, news, and market conditions
        """
        # Current NVDA data (as of implementation date)
        # In a real system, this would come from financial APIs
        mock_data = {
            'stock_data': {
                'ticker': 'NVDA',
                'current_price': 157.75,
                'previous_close': 155.02,
                'day_change': 2.73,
                'day_change_percent': 1.76,
                'week_change_percent': 8.46,
                'month_change_percent': 15.97,
                'year_change_percent': 27.12,
                'volume': 52_400_000,
                'market_cap': 3.85e12,  # $3.85 trillion
                'week_52_low': 86.62,
                'week_52_high': 158.71,
                'avg_volume_3m': 48_200_000
            },
            'news_data': [
                {
                    'headline': 'NVIDIA Becomes World\'s Most Valuable Company After AI Surge',
                    'sentiment': 'positive',
                    'impact': 'high',
                    'summary': 'NVDA surpassed Microsoft to become the most valuable company globally.'
                },
                {
                    'headline': 'Strong Demand for AI Chips Drives NVIDIA Revenue Growth',
                    'sentiment': 'positive', 
                    'impact': 'medium',
                    'summary': 'Data center revenue continues to show exceptional growth.'
                },
                {
                    'headline': 'Analysts Raise NVIDIA Price Targets on AI Optimism',
                    'sentiment': 'positive',
                    'impact': 'medium',
                    'summary': 'Multiple investment firms increased target prices following earnings.'
                }
            ],
            'market_conditions': {
                'overall_market': 'bullish',
                'sector_performance': 'technology_leading',
                'volatility_index': 15.2,  # VIX equivalent
                'ai_sector_sentiment': 'very_positive',
                'economic_indicators': 'stable'
            }
        }
        
        return mock_data
    
    def _create_agents(self) -> Dict[str, ConversableAgent]:
        """
        Create and configure the four specialized agents.
        
        This demonstrates AG2's modern ConversableAgent creation pattern
        using the LLMConfig context manager for proper configurati    
        """
    

        if self.verbose:
            print("🤖 Creating specialized agents...")
        
        agents = {}
        
        # Use the modern AG2 pattern with LLMConfig as context manager
        with self.llm_config:
            # 1. Data Analyst Agent
            # Specializes in quantitative analysis of stock data
            agents['data_analyst'] = ConversableAgent(
                name="data_analyst",
                system_message="""You are a Data Analyst Agent specializing in quantitative stock analysis.

Your role is to:
1. Analyze price data and calculate key metrics
2. Identify trends and patterns in stock performance
3. Provide clear, data-driven insights
4. Present findings in an organized, educational format

Focus on metrics like:
- Price changes (daily, weekly, monthly)
- Volume analysis
- Volatility assessment
- Performance relative to historical ranges

Always explain your calculations and reasoning to help readers understand
the analytical process.""",
                human_input_mode="NEVER"
            )
            
            # 2. News Analyst Agent  
            # Specializes in sentiment analysis and news impact assessment
            agents['news_analyst'] = ConversableAgent(
                name="news_analyst",
                system_message="""You are a News Analyst Agent specializing in financial news sentiment analysis.

Your role is to:
1. Analyze news headlines and content for market sentiment
2. Assess potential impact of news on stock performance
3. Identify key themes and market narratives
4. Provide sentiment scoring and impact assessment

Focus on:
- Overall sentiment (positive, negative, neutral)
- Impact assessment (high, medium, low)
- Key themes and catalysts
- Market narrative analysis

Present your findings clearly and explain how news sentiment
typically affects stock performance.""",
                human_input_mode="NEVER"
            )
            
            # 3. Risk Assessment Agent
            # Specializes in risk evaluation and market condition analysis
            agents['risk_analyst'] = ConversableAgent(
                name="risk_analyst", 
                system_message="""You are a Risk Assessment Agent specializing in market risk evaluation.

Your role is to:
1. Evaluate various risk factors affecting the stock
2. Assess market conditions and volatility
3. Identify potential threats and opportunities
4. Provide risk-adjusted perspective on the investment

Focus on:
- Market volatility and stability
- Sector-specific risks
- Economic environment impact
- Technical risk indicators
- Overall risk profile assessment

Provide balanced analysis that helps investors understand
both potential rewards and risks.""",
                human_input_mode="NEVER"
            )
            
            # 4. Coordinator Agent
            # Synthesizes findings from all specialists into final report
            agents['coordinator'] = ConversableAgent(
                name="coordinator",
                system_message="""You are the Coordinator Agent responsible for synthesizing analysis from all specialists.

Your role is to:
1. Integrate findings from Data Analyst, News Analyst, and Risk Analyst
2. Create a comprehensive, coherent final report
3. Highlight key insights and recommendations
4. Present information in a clear, actionable format

Your final report should include:
- Executive summary of key findings
- Data analysis highlights
- News sentiment impact
- Risk assessment summary
- Overall outlook and considerations

Structure your response to be informative for both novice and
experienced investors. Format your response in the markdown format.

Use professional, objective language. Avoid emotional language. Never use emojis.
If you have numerical data present it formatting it accordingly and only then provide your assessment.

Follow this format:

# Executive Summary

# Data Analysis Highlights

# News Sentiment Impact

# Risk Assessment Summary

# Recommendations

""",
                human_input_mode="NEVER"
            )
        
        if self.verbose:
            print(f"✅ Created {len(agents)} specialized agents")
            
        return agents

    def _prepare_analysis_prompt(self, ticker: str) -> str:
        """Prepare the initial analysis prompt with mock data."""
        stock_data = self.mock_data['stock_data']
        
        prompt = f"""Please analyze {ticker} stock with the following current data:

STOCK DATA:
- Current Price: ${stock_data['current_price']:.2f}
- Previous Close: ${stock_data['previous_close']:.2f}  
- Day Change: ${stock_data['day_change']:.2f} ({stock_data['day_change_percent']:.2f}%)
- Week Change: {stock_data['week_change_percent']:.2f}%
- Month Change: {stock_data['month_change_percent']:.2f}%
- Year Change: {stock_data['year_change_percent']:.2f}%
- Volume: {stock_data['volume']:,}
- Market Cap: ${stock_data['market_cap']:.2e}
- 52-Week Range: ${stock_data['week_52_low']:.2f} - ${stock_data['week_52_high']:.2f}

Please provide your specialized analysis based on this data."""
        
        return prompt
    
    def _format_news_data(self) -> str:
        """Format mock news data for agent analysis."""
        news_items = []
        for item in self.mock_data['news_data']:
            news_items.append(f"• {item['headline']}")
            news_items.append(f"  Summary: {item['summary']}")
            news_items.append(f"  Sentiment: {item['sentiment']}, Impact: {item['impact']}")
            news_items.append("")
        
        return "\n".join(news_items)
    
    def _format_market_conditions(self) -> str:
        """Format mock market conditions for agent analysis."""
        conditions = self.mock_data['market_conditions']
        
        formatted = f"""MARKET CONDITIONS:
- Overall Market: {conditions['overall_market']}
- Sector Performance: {conditions['sector_performance']}
- Volatility Index: {conditions['volatility_index']}
- AI Sector Sentiment: {conditions['ai_sector_sentiment']}
- Economic Indicators: {conditions['economic_indicators']}"""
        
        return formatted
    
    def analyze_stock(self, ticker: str = "NVDA") -> str:
        """
        Orchestrate the multi-agent stock analysis using AG2's proper sequential chat pattern.
        
        This method demonstrates AG2's initiate_chats() method where:
        1. Each chat's summary becomes carryover for the next
        2. Framework handles the sequential orchestration
        3. Agents automatically build upon previous analysis
        
        Args:
            ticker: Stock symbol to analyze (defaults to NVDA)
            
        Returns:
            str: Final comprehensive analysis report
        """
        if self.verbose:
            print(f"\n📊 Starting multi-agent analysis for {ticker}")
            print("=" * 60)
        
        # Prepare initial analysis prompt with mock data
        initial_prompt = self._prepare_analysis_prompt(ticker)
        
        # Create a coordinator user proxy to orchestrate the sequential chat
        user_proxy = UserProxyAgent(
            name="analysis_coordinator",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0,
            system_message="You are coordinating a stock analysis workflow."
        )
        
        # Configure sequential chat workflow using AG2
        chat_results = None
        with self.llm_config:
            if self.verbose:
                print("\n🔄 Initiating AG2 sequential chat workflow...")
            
            chat_results = user_proxy.initiate_chats([
                # Step 1: Data Analysis
                {
                    "recipient": self.agents['data_analyst'],
                    "message": initial_prompt,
                    "max_turns": 1,
                    "summary_method": "last_msg",
                    "summary_prompt": "Summarize the key data analysis findings."
                },
                # Step 2: News Sentiment Analysis (with carryover from data analysis)
                {
                    "recipient": self.agents['news_analyst'], 
                    "message": f"Now analyze the news sentiment for {ticker}. Consider the previous data analysis.\n\nNews data:\n{self._format_news_data()}",
                    "max_turns": 1,
                    "summary_method": "last_msg",
                    "summary_prompt": "Summarize the key news sentiment findings."
                },
                # Step 3: Risk Assessment (with carryover from both previous analyses)
                {
                    "recipient": self.agents['risk_analyst'],
                    "message": f"Evaluate risk factors for {ticker}. Consider the previous data and news analysis.\n\nMarket conditions:\n{self._format_market_conditions()}",
                    "max_turns": 1,
                    "summary_method": "last_msg", 
                    "summary_prompt": "Summarize the key risk assessment findings."
                },
                # Step 4: Final Synthesis (with carryover from all previous analyses)
                {
                    "recipient": self.agents['coordinator'],
                    "message": f"Create a comprehensive final analysis report for {ticker}, synthesizing all previous specialist findings.",
                    "max_turns": 1,
                    "summary_method": "last_msg"
                }
            ])
        
        # Extract final report - simplified for educational clarity
        final_report = "Analysis completed successfully."
        if chat_results and chat_results[-1].summary:
            final_report = chat_results[-1].summary
        
        if self.verbose:
            print("\n✅ AG2 sequential chat workflow complete!")
            print("=" * 60)
        
        return final_report
    
    


def main():
    """
    Main execution function demonstrating the complete stock analysis workflow.
    
    This function shows how to:
    1. Initialize the multi-agent system
    2. Run a complete stock analysis
    3. Display results in an educational format
    """
    print("🚀 AG2 Stock Analysis Agent System - Part 2")
    print("Educational Multi-Agent Implementation")
    print("=" * 60)
    
    try:
        # Initialize the stock analysis system
        print("⚙️ Initializing multi-agent system...")
        analysis_system = StockAnalysisSystem()
        
        # Run stock analysis for NVDA
        print(f"\n📈 Analyzing NVDA using {len(analysis_system.agents)} specialized agents...")
        
        # Execute the multi-agent analysis
        final_report = analysis_system.analyze_stock("NVDA")
        
        # Display final results
        print("\n" + "=" * 60)
        print("📊 FINAL ANALYSIS REPORT")
        print("=" * 60)
        print(final_report)
        print("\n" + "=" * 60)
        print("✅ Analysis complete! This demonstrates AG2's sequential chat orchestration.")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        logger.error(f"Main execution error: {e}", exc_info=True)


if __name__ == "__main__":
    main()