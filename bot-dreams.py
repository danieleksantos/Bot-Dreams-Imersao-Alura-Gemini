# Instalar Framework de agentes do Google ################################################
!pip install -q google-adk

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types  # Para criar conteúdos (Content e Part)
from datetime import date
import textwrap # Para formatar melhor a saída de texto
from IPython.display import display, Markdown # Para exibir texto formatado no Colab
import requests # Para fazer requisições HTTP
import warnings

warnings.filterwarnings("ignore")

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response

    # Função auxiliar para exibir texto formatado em Markdown no Colab
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

  ##########################################
# --- Agente 1: Buscador de Referências de Freud --- #
##########################################
def agente_buscador(sonho):

    buscador = Agent(
        name="agente_buscador",
        model="gemini-2.0-flash",
        instruction="""
        Você agora vai atuar como se fosse o Freud. A sua tarefe é usar a ferramenta de busca do google (google_search)
        para encontrar toda a bibliografia disponível de Freud e recuperar o que for relevantes sobre o tópico abaixo,
        principalmente referente aos sonhos. Você irá converter todo o conteúdo em uma linguagem amigável para
        leigos em psicologia e psquiatria.
        """,
        description="Agente que busca referencias no Google",
        tools=[google_search]
    )

    entrada_do_agente_buscador = f"Relate seu sonho: {sonho}"

    conteudo = call_agent(buscador, entrada_do_agente_buscador)
    return sonho

    ################################################
# --- Agente 2: Planejador do resultado --- #
################################################
def agente_planejador(sonho, referencia_buscada):
    planejador = Agent(
        name="agente_planejador",
        model="gemini-2.0-flash",
        # Inserir as instruções do Agente Planejador #################################################
        instruction="""
        Você é um psicológo pesquisador especialista em Freud e sonhos. Com base na referencia
        buscada anteriormente e outras referencias relevantes da ferramenta de busca do Google (google_search) você deve:
        interpretar o sonho que o usuário relatou e se aprofundar mais nesse tema.
        Ao final, você irá relatar a sua análise sobre o sonho com base em todas as referencias.
        """,
        description="Agente que planeja a resposta da interpretação dos sonhos",
        tools=[google_search]
    )

    entrada_do_agente_planejador = f"Sonho:{sonho}\nReferencia buscada: {referencia_buscada}"
    # Executa o agente
    interpretacao_do_sonho = call_agent(planejador, entrada_do_agente_planejador)
    return interpretacao_do_sonho


    ######################################
# --- Agente 3: Redator da interpretação do sonho --- #
######################################
def agente_redator(sonho, interpretacao_do_sonho):
    redator = Agent(
        name="agente_redator",
        model="gemini-2.0-flash",
        instruction="""
            Agora você é uma versão moderna e amável de Freud. Você vai usar as referencias fornecidas
            e o sonho do usuário para contar a ele de uma forma simples e usando linguagem acessivel o
            que o sonho dele significa segundo os estudos e freud.
            Utilize o tema fornecido no sonho e na interpretação do sonho, assim como e os pontos mais
            relevantes fornecidos e, com base nisso, escreva um rascunho resposta ao usuário.
            A resposta deve ser um tom leve e divertido e o usuário deverá ser lembrado de que esse bot,
            apesar de se basear em referencias reais é apenas um objeto de de estudo e não substitui
            qualquer consulta com o profissonal psicologo.
            """,
        description="Agente redator da interpretação do sonho"
    )
    entrada_do_agente_redator = f"Sonho:{sonho}\nReferencia buscada: {interpretacao_do_sonho}"
    # Executa o agente
    rascunho_resposta = call_agent(redator, entrada_do_agente_redator)
    return rascunho_resposta


    ##########################################
# --- Agente 4: Revisor de Qualidade --- #
##########################################
def agente_revisor(sonho, resposta_gerada):
    revisor = Agent(
        name="agente_revisor",
        model="gemini-2.0-flash",
        instruction="""
            Você é um freud que vive em 2025, adequeque a interpretação do sonho ao contexto
            atual e principalmente ao público jovem, entre 18 e 30 anos, use um tom de escrita informal, leve e descontraído.
            Revise o rascunho da resposta, verificando clareza, concisão, correção e tom.
            Se o rascunho estiver bom, responda apenas 'A interpretação do sonho foi concluída!'.
            Finalize com um conselho final  específico e relevante para o contexto do sonho.
            Remova frases repetitivas ou explicativas demais.
            Caso haja problemas, aponte-os e sugira melhorias.
            """,
        description="Agente revisor da resposta final."
    )
    entrada_do_agente_revisor = f"Sonho:{sonho}\nRascunho: {resposta_gerada}"
    # Executa o agente
    sonho_interpretado = call_agent(revisor, entrada_do_agente_revisor)
    return sonho_interpretado


    
print("🚀 Iniciando o Sistema de Intepretação dos sonhos com 4 Agentes 🚀")

# --- Obter o sonho do Usuário ---
sonho = input("❓ Por favor, me conte o seu sonho que Freud irá interpretar: ")

# Inserir lógica do sistema de agentes ################################################
if not sonho:
    print("Você esqueceu de digitar o sonho!")
else:
    print(f"Maravilha! Vamos então interpretar seu sonho!")

    referencia_buscada = agente_buscador(sonho)
    print("\n--- 📝 Resultado do Agente 1 (Buscador) ---\n")
    display(to_markdown(referencia_buscada))
    print("--------------------------------------------------------------")

    interpretacao_do_sonho = agente_planejador(sonho, referencia_buscada)
    print("\n--- 📝 Resultado do Agente 2 (Planejador) ---\n")
    display(to_markdown(interpretacao_do_sonho))
    print("--------------------------------------------------------------")

    rascunho_resposta = agente_redator(sonho, interpretacao_do_sonho)
    print("\n--- 📝 Resultado do Agente 3 (Redator) ---\n")
    display(to_markdown(rascunho_resposta))
    print("--------------------------------------------------------------")

    sonho_interpretado = agente_revisor(sonho, rascunho_resposta)
    print("\n--- 📝 Resultado do Agente 4 (Revisor) ---\n")
    display(to_markdown(sonho_interpretado))
    print("--------------------------------------------------------------")