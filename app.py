import mimetypes
from math import ceil
from typing import List

import pathlib
import pandas as pd
import numpy as np
from shiny import App, Inputs, Outputs, Session, render, ui,reactive
from shiny.types import FileInfo
from shiny.types import NavSetArg
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho

import matplotlib.pyplot as plt

dir = pathlib.Path(__file__).parent #permite obtener la dirección relativa

#sección de código de shiny, interfaz de usuario
def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        ui.nav("Cargar Archivo", prefix + " Dataframe de Portafolio", #ópción para cargar el archivo excel
               ui.page_fluid(

    ui.layout_sidebar(
        ui.panel_sidebar(
             ui.input_file("file1", "Selecciona un archivo en formato de Excel", accept=[".xlsx"], multiple=False),
            ui.input_checkbox("header", "Nombre Columas", True),           

        ),
        ui.panel_main(
                      ui.output_ui("contents"), #etiqueta para mostrar la información subida del archivo excel
                      
                      )
    )    
)),
        ui.nav("Black Scholes", prefix + " Modelo Black Scholes",#opción Black-Shcoles
               ui.page_fluid(
                            
    ui.layout_sidebar(
        ui.panel_sidebar( 
        ui.markdown(  #etiqueta para mostrar el total de bs_price
        """      
        Total **bs_price**:              
        """
    ),         
           ui.output_text_verbatim("blackScholescontentsSuma"),   #etiqueta para mostar el cálculo de portafolio                        
        ),
        ui.panel_main(
        ui.markdown( #etiqueta para mostrar los tipos c y p
        """      
        Tipo **c**:               
        """
    ),
                      ui.output_ui("blackScholescontentsC"),#etiqueta para mostrar los tipo c
                      ui.markdown( 
        """      
        Tipo **p**:              
        """
    ),
                       ui.output_ui("blackScholescontentsP"), #etiqueta para mostrar los tipo p
                       ui.output_plot("plot"), # etiqueta para mostrar el gráfico del portafolio
                      )
    )    
)),
         ui.nav("Análisis de Sensibilidad", prefix + ": Análisis de Sensibilidad", #etiqueta para mostrar el análisis de sensibilidad
                 ui.page_fluid(
                            
    ui.layout_sidebar(
        ui.panel_sidebar(         
                     ui.input_numeric("Sigma", "Sigma:", 0, min=0, max=100), #cuadros de texto numéricos
                     ui.input_numeric("S", "S:", 0, min=0, max=100),                   
                         
        ),
        ui.panel_main(ui.output_ui("sensibilidad"),
                   ui.output_plot("plot_sensibilidad"),)
                   # ui.output_ui("sensibilidadMatriz"),  
                      
    )    
)),
        ui.nav_spacer(),
       
    ]

app_ui = ui.page_navbar(  #permite mostrar las opciones, es la parte principal de la pantalla
    *nav_controls("Contenido: "),
    title="Valoración de opciones",
    bg="#0062cc", #color de menú principal
    inverse=True,
    id="navbar_id",
    footer=ui.div(
        {"style": "width:80%;margin: 0 auto"},
        ui.tags.style(
            """
            h4 {
                margin-top: 3em;
            }
            """
        ),
       
    )
)
#sección del código python
def server(input, output, session): 
    #permite cargar los archivos temporales de manera reactiva
    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())    
    #df = pd.read_excel('archivo.xlsx')

    @reactive.file_reader(dir / "archivo.xlsx")
    def read_file():
        #df = pd.read_excel('archivo.xlsx')
        return pd.read_excel(dir / "archivo.xlsx")
    
    @reactive.file_reader(dir / "results_sensitivity.xlsx")
    def read_file_sensitivity():        
        return pd.read_excel(dir / "results_sensitivity.xlsx")
    
    @reactive.file_reader(dir / "blackScholesC.xlsx")
    def read_file_typec():        
        return pd.read_excel(dir / "blackScholesC.xlsx")
    
    @reactive.file_reader(dir / "blackScholesP.xlsx")
    def read_file_typep():        
        return pd.read_excel(dir / "blackScholesP.xlsx")
        
    @reactive.file_reader(dir / "blackScholesS.xlsx")
    def read_file_types():        
        return pd.read_excel(dir / "blackScholesS.xlsx")  
      
    @reactive.file_reader(dir / "sensibilidad_data.xlsx")
    def read_file_sensibilidad_data():        
        return pd.read_excel(dir / "sensibilidad_data.xlsx")  

    @output   
    @render.ui
    def contents():
        if input.file1() is None:
            return "Sube un archivo en xlsx"  #permite cargar el archivo del portafolio
        f: list[FileInfo] = input.file1()
        df = pd.read_excel(f[0]["datapath"], header=0 if input.header() else None)                
        df.to_excel('archivo.xlsx', header=True, index=False)#guarda de manera temporal la información 
        
        pd.DataFrame(columns = ['S','price']).to_excel('sensibilidad_data.xlsx', index=False, header=True)
        
        return ui.HTML(df.to_html(classes="table table-striped"))# permite desplegar el archivo en una lista
    
    @output
    @render.ui
    def blackScholescontentsC():   #función que ayuda a filtrar los tipo c     
        df = read_file()        #lee el archivo temporal con la información que se subió del portafolio
        df = df[df['type'] == 'c'] #filtra del DataFrame los tipo c
        results = df.apply(lambda row: blackScholes(row['Nombre del contrato'],row['r'], row['S'], row['K'], row['T'], row['sigma'], row['type']), axis=1)
        final_df = pd.concat(results.to_list(), ignore_index=True)          
        final_df.to_excel('blackScholesC.xlsx', header=True, index=False) #guarda temporalmente la información tipo c
        return ui.HTML(final_df.to_html(classes="table table-striped"))# muestra la lista de los tipo c
    
    @output
    @render.ui
    def blackScholescontentsP():      #función que ayuda a filtrar los tipo p  
        df = read_file()         #lee el archivo temporal con la información que se subió del portafolio
        df = df[df['type'] == 'p']  #filtra del DataFrame los tipo p
        results = df.apply(lambda row: blackScholes(row['Nombre del contrato'],row['r'], row['S'], row['K'], row['T'], row['sigma'], row['type']), axis=1)
        final_df = pd.concat(results.to_list(), ignore_index=True)          
        final_df.to_excel('blackScholesP.xlsx', header=True, index=False)  #guarda temporalmente la información tipo c
        dfS = read_file()       #lee el archivo temporal con la información que se subió del portafolio
        dfS=dfS['S'] #filtra el valor S
        dfS.iloc[-1] #  obtiene el valor S
        dfS.to_excel('blackScholesS.xlsx', header=True, index=False) #guarda temporalmente el valor S
        return ui.HTML(final_df.to_html(classes="table table-striped"))# muestra la lista de los tipo c
      
    
    @output
    @render.text
    def blackScholescontentsSuma(): #función que permite optener el total del portafolio       
        dfC = read_file_typec() #Lee lo archivos temporales tipo c
        dfP = read_file_typep() #Lee lo archivos temporales tipo p
        dfC = float(dfC['bs_price'].sum())#obtine la suma de los tipo c
        dfP = float(dfP['bs_price'].sum())#obtine la suma de los tipo p       
        total=dfC-dfP  #calcula el total c-p
        return total        

    @output
    @render.plot(alt="Representación de la cartera") #función que permite representar gráficamente el portafolio
    def plot():
        dfC = read_file_typec() #Lee lo archivos temporales tipo c
        dfS = read_file_types() #Lee lo archivos temporales tipo s
        dfP = read_file_typep() #Lee lo archivos temporales tipo 
        total_price = int(dfC['bs_price'].sum()) - int(dfP['bs_price'].sum()) #TotalPrice          
        dfS=dfS['S'] #  obtiene el valor S
        S=int(dfS.iloc[0]) # obtiene el valor S
        
        # Crear la figura y los ejes
        fig, ax = plt.subplots()
        # Dibujar puntos
        ax.scatter(x = S, y = total_price)# x=S y=C
        ax.set_xlabel("Precio del activo subyasente (S)", fontdict = {'fontsize':10, 'fontweight':'bold', 'color':'tab:blue'}) #da formato al gráfico
        ax.set_ylabel("Valor del portafolio (Total bs_price)", loc = "bottom", fontdict = {'fontsize':10, 'fontweight':'bold', 'color':'tab:blue'})
        ax.set_title('Representación gráfica del portafolio', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
        return fig    
        
        
    @output
    @render.plot(alt="Representación de la simulación del portafolio") #función que permite representar gráficamente el portafolio
    def plot_sensibilidad():
        df = read_file_sensibilidad_data()
        # Crear la figura y los ejes
        fig, ax = plt.subplots()
        # Dibujar puntos
        ax.plot(df['S'], df['price'], marker='o', linestyle='-', color='tab:orange')        
        ax.set_xlabel("Precio del activo subyasente (S)", fontdict = {'fontsize':10, 'fontweight':'bold', 'color':'tab:blue'}) #da formato al gráfico
        ax.set_ylabel("Valor del portafolio (Total bs_price)", loc = "bottom", fontdict = {'fontsize':10, 'fontweight':'bold', 'color':'tab:blue'})
        ax.set_title('Representación gráfica del portafolio', loc = "left", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
        return fig 
    
    
    
    #función para el cálculo de Black-Scholes
    def blackScholes(Nombredelcontrato,r, S, K, T, sigma, type="c"): #recibe los valores enviados
        
        bs_price = bs(type, S, K, T, r, sigma)
        delta_calc = delta(type, S, K, T, r, sigma)
        gamma_calc = gamma(type, S, K, T, r, sigma)
        vega_calc = vega(type, S, K, T, r, sigma)
        theta_calc = theta(type, S, K, T, r, sigma)*(365/252)
        rho_calc = rho(type, S, K, T, r, sigma)
    
        data = [[Nombredelcontrato,bs_price, delta_calc, gamma_calc, vega_calc, theta_calc, rho_calc]]

        df = pd.DataFrame(data, columns=['Nombredelcontrato','bs_price', 'delta_calc', 'gamma_calc', 'vega_calc', 'theta_calc', 'rho_calc'])
    
        return df #regresa el valor calculado
    
    @output  
    @render.ui
    def sensibilidad():        
        df = read_file()
        sigma_value=input.Sigma()
        s_value=input.S()
        results_sensitivity = pd.DataFrame(columns = ['Nombre del contrato', 'delta_calc', 'gamma_calc', 'vega_calc', 'theta_calc', 'rho_calc'])
        df_temp = df.copy()  # Crear una copia del DataFrame original
#             Actualizar los parámetros en el DataFrame temporal
        df_temp['sigma'] = (sigma_value/100 )+df_temp['sigma']
        df_temp['S'] = s_value + df_temp['S']
#             Calcular nuevamente el precio y las métricas de las opciones
        simulacion = df_temp.apply(lambda row: blackScholes(row['Nombre del contrato'],row['r'], row['S'], row['K'], row['T'], row['sigma'], row['type']), axis=1) 
        df_temp[['Nombre del contrato','bs_price', 'delta_calc', 'gamma_calc', 'vega_calc', 'theta_calc', 'rho_calc']] = pd.concat(simulacion.to_list(), ignore_index=True)
        # Agregar los resultados al análisis de sensibilidad
        results_sensitivity = pd.concat([results_sensitivity,df_temp])
        results_sensitivity.to_excel('results_sensitivity.xlsx', header=True, index=False)
        data_plot = results_sensitivity.copy()
        data_plot.loc[data_plot['type']=='p','bs_price'] = data_plot['bs_price'] * -1
        price = int(data_plot['bs_price'].sum()) #obtine la suma de los tipo c        
        S = int(data_plot.iloc[0]['S'])
        data_plot_aux = pd.DataFrame(data = {'S':[S], 'price':[price]})
        data_plot_aux2 = read_file_sensibilidad_data()
        pd.concat([data_plot_aux, data_plot_aux2]).to_excel('sensibilidad_data.xlsx', index = False, header = True)
        return ui.HTML(results_sensitivity[['Nombre del contrato','type','bs_price', 'delta_calc', 'gamma_calc', 'vega_calc', 'theta_calc', 'rho_calc']].to_html(classes="table table-striped"))
        

      


    



app = App(app_ui, server)
