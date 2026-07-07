import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")
    information = """
    Ariana Grande-Butera (born June 26, 1993) is an American singer, songwriter, and actress. Known for her four-octave vocal range, which extends into the whistle register, she is an influential figure in popular music. Publications such as Rolling Stone and Billboard have deemed Grande one of the greatest artists in history, while Time included her on its list of the world's 100 most influential people in 2016 and 2019.

Grande's career began as a teenager in the Broadway musical 13 (2008) before she gained prominence as Cat Valentine in the Nickelodeon television series Victorious (2010 to 2013) and its spin-off Sam & Cat (2013–2014). After signing with Republic Records, she released her debut studio album, Yours Truly (2013), a retro-inspired pop and R&B record that debuted atop the Billboard 200. She incorporated elements of electronic on her next two albums, My Everything (2014) and Dangerous Woman (2016), which both achieved international success, spawning the hit singles "Problem", "Break Free", "Bang Bang", "One Last Time", "Into You", and "Side to Side".

Personal struggles influenced Grande's albums Sweetener (2018) and Thank U, Next (2019), both of which delved into trap. The latter garnered the US Billboard Hot 100 number-one singles "Thank U, Next" and "7 Rings". With the title track of her R&B-infused album Positions (2020), as well as the collaborations "Stuck with U" and "Rain on Me", she achieved the most number-one debuts in the US. After a musical hiatus, she explored dance on Eternal Sunshine (2024), which yielded the US number-one songs "Yes, And?" and "We Can't Be Friends (Wait for Your Love)". She returned to acting with the political satire Don't Look Up (2021) and portrayed Glinda in the fantasy musical film Wicked (2024), which earned her an Academy Award nomination, as well as its sequel Wicked: For Good (2025). Her upcoming eighth album, Petal (2026), contains her tenth US number-one single, "Hate That I Made You Love Me".

Grande is one of the best-selling music artists ever, with estimated sales of over 90 million records. The highest-paid female musician in 2020, her accolades include three Grammy Awards, a Brit Award, two Billboard Music Awards, three American Music Awards, forty Guinness World Records, and thirteen MTV Video Music Awards. Grande has amassed six Billboard 200 number-one albums and ten Billboard Hot 100 number-one singles. Outside of music and acting, she has worked with many charitable organizations and advocates for animal rights, mental health, and gender, racial, and LGBT equality. Her business ventures include the cosmetics brand R.E.M. Beauty and a fragrance line that has earned over $1 billion in global retail sales. She has a large social media following, being the sixth-most-followed individual on Instagram.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm_openai = ChatOpenAI(
        model="gpt-5",
        temperature=0,
    )

    llm_gemma3 = ChatOllama(
        model="gemma3:270m",
        temperature=0,
    )

    chain = summary_prompt_template | llm_openai
    response = chain.invoke(input = {"information": information})

    print(response.content)

if __name__ == "__main__":
    main()
