import cv2
import time
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

    ##### RASPI #####

from picamera.array import PiRGBArray
from picamera import PiCamera
from gpiozero import LED #PODER ACCIONAR LOS GPIO
from subprocess import call

    ##### VENTANA #####

import wx
import gettext
import threading
import joblib

##### HILO PRINCIPAL #####  
      
class hilovalores(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.runnig = True
        
        self.tiempo_respuesta = None
        self.area_relativa = None
        
        
    ##### INICIALIZA LA CAMARA Y CONFIGURA #####
            
        self.camera = PiCamera()
        self.camera.resolution = (1280, 720) #SE CONFIGURA LA RESOLUCIóN EN 720P
        self.camera.framerate = 60 #SE CONFIGURA LA LOS FRAMES EN 60
        
    ##### LEDS #####
        
        self.led = LED(6) #LED BLANCO
        self.ledIR = LED(16) #LED INFRAROJO
        self.ledIR.on() #ENCIENDE EL LED INFRAROJO
        
    #SE CALIENTA LA CAMARA
        
        self.camera.start_preview() 
        time.sleep(2)
        self.camera.stop_preview()
        
    ##### DEFINICION DE FUNCION DE PORCENTAJE ######
        
    def run(self):
        
    ##### DECLARACION VARIABLES #####
    
        self.tiempo_frames = pd.DataFrame()
        self.m=0 # VARIABLE UTILIZADA PARA LA UBICACIÓN DEL AREA A CARGAR 
        self.tiempo_inicial = None # SE DECALRA COMO VACIA PARA QUE INGRESE SOLO UNA VEZ EN EL BUCLE
        self.frame_inicial = None # SE DECALRA COMO VACIA PARA QUE INGRESE SOLO UNA VEZ EN EL BUCLE
        self.tiempo_final = None # SE DECALRA COMO VACIA PARA QUE INGRESE SOLO UNA VEZ EN EL BUCLE
        self.area_ellipse = None

        self.mediana = np.array([]) #SE DECLARA VARIABLE COMO UN ARRAY
        self.mediana_derivada = np.array([]) #SE DECLARA VARIABLE COMO UN ARRAY
        self.dadt = np.array([]) #SE DECLARA VARIABLE COMO UN ARRAY
        self.area_minima_2 = 1000000
        self.contador = 0
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        
    ##### NOMBRE DEL VIDEO #####
        
        f = open('nombre.txt','r')
        nombre = f.read()
        nombre = int(nombre) +1
        f.close()
        
        f = open ('nombre.txt','w')
        f.write(str(nombre))
        f.close()
        
    ##### SE GRABA VIDEO #####
        
        self.camera.start_recording(f"test_{nombre}.h264") #COMIENZA A GRABAR
        self.led.on() # ENCIENDE EL LED BLANCO
        time.sleep(0.2) # ESPERA 0.2 SEGUNDOS
        self.led.off() # APAGA LED BLANCO
        time.sleep(1.5) # ESPERA 1 SEGUNDO
        self.camera.stop_recording() #TERMINA DE GRABAR
        self.ledIR.off() #APAGA EL LED INFRAROJO
        
    ##### CONVIERTE EL VIDEO EN MP4 #####
        
        command = f"MP4Box -add test_{nombre}.h264 test_{nombre}.mp4"#Define el comando comando que va a ejecutarse
        call([command], shell=True)
        
    ##### CAPTURA DEL VIDEO A ANALIZAR Y DE SUS PARÁMETROS #####
    
        self.video = cv2.VideoCapture(f"test_{nombre}.mp4") # CAPTURA VIDEO #CONTRACCION EN EL FRAME 72/73
        self.frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) # CALCULA FRAMES EN VIDEO
        self.fps = 60  
        self.tiempo_frames = 1 / self.fps # CALCULA TIEMPO ENTRE FRAMES
        self.parametros = pd.DataFrame(columns=['Frame','Tiempo','Area'],index=range(self.frames)) # CREA UN DATAFRAME CON 2 COLUMNAS CON ESOS NOMBRES Y UNA CANTIDAD DE FILAS IGUAL AL NUMERO DE FRAMES
        self.parametros_luz = pd.DataFrame(columns=['Frame','Tiempo','Area'])
 
        for i in range(self.frames): #BUCLE QUE RELLENA EL DATAFRAME CON EL NUMERO DE FRAME, TIEMPO TRANSCURRIDO Y EL AREA EN 0"
            self.parametros.iloc[i] = (i+1,self.tiempo_frames*i,'0',)    
    
        self.tiempo_constante = pd.DataFrame(np.zeros((self.frames, 1)))
        self.run_loop = True
        
        while self.run_loop: #REPITE EL LOOP POR CADA FRAME
            ret, self.frame = self.video.read()
            
            if ret is False:
                break
    
    ##### 6.2.4. PROCESAMIENTO DE IMÁGENES #####
            
    ##### REGION DE INTEREZ #####   

            self.roi = self.frame[0:500 , 350:850] # [FILAS,COLUMNAS] SETEADO PARA: casco_720p60prueba2.mp4
        
    ##### TRANSFORMA ROI A ESCALA DE GRISES #####
        
            self.roi_gris = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
        
    ##### UMBRALIZACIÓN #####
        
            ret2, self.umbralizada = cv2.threshold(self.roi_gris, 120, 255,cv2.THRESH_TOZERO_INV)
        
    ##### OPERACIONES MORFOLÓGICAS#####
        
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    
            self.dilatada_erosionada = cv2.erode(cv2.dilate(self.umbralizada,kernel,iterations=15),kernel,iterations=15)
    
    ##### 6.2.5. EXTRACCIÓN DE PARÁMETROS #####
                    
            self.contornos, hierarchy = cv2.findContours(self.dilatada_erosionada, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
            self.drawing = self.roi
            cv2.drawContours(self.drawing, self.contornos, -1, (255, 0, 0), 1)
            
            for self.contornos in self.contornos:
                self.contornos = cv2.convexHull(self.contornos) # ENCUENTRA LA ENVOLVENTE CONVEXA DE UN CONJUNTO DE PUNTOS, EN ESTE CASO DE LOS CONTORNOS
                self.area_contorno = cv2.contourArea(self.contornos) # CALCULA EL AREA DE LA ENVOLVENTE
                self.bounding_box = cv2.boundingRect(self.contornos)  # CALCULA LOS LÍMITES DEL RECTANGULO CONTENEDOR DE LOS PUNTOS
                self.extension = self.area_contorno / (self.bounding_box[2] * self.bounding_box[3]) # CALCULA LA RELACIÓN ENTRE EL AREA DE LA ENVOLVENTE Y EL RECTANGULO CONTENEDOR
    
                if self.extension > 0.8: # SE CONTINÚA SOLO CON LAS AREAS QUE POSEAN UNA RELACIÓN DE 0.8 CON SU RECTANGULO ENVOLVENTE
                    continue
                
                if self.area_contorno < 1800: # SI EL AREA DEL CONTORNO ES MENOR A 1800 SE DESCARTA
                    continue
                
                if self.area_contorno > 30000: # SI EL AREA DEL CONTORNO ES MAYOR A 30000 SE DESCARTA
                    continue
            
                self.circunferencia = cv2.arcLength(self.contornos,True) # SE CALCULA LA CIRCUNFERENCIA
                self.circularidad = self.circunferencia ** 2 / (4*np.pi*self.area_contorno) # SE CALCULA CIRCULARIDAD
                
                if self.circularidad > 1.4: # SE DESCARTAN AREAS CON UNA CIRCULARIDAD MAYOR A 1.4
                    continue
            
                self.g = cv2.moments(self.contornos) # SE ENCUENTRA EL CENTRO DEL CONTORNO MEDIANTE MOMENTOS 
                if self.g['m00'] != 0:
                    self.center = (int(self.g['m10'] / self.g['m00']), int(self.g['m01'] / self.g['m00']))
                    cv2.circle(self.drawing, self.center, 3, (0, 255, 0), -1)
            

                try: #TRATA DE EJECUTAR LO QUE SIGUE, SI SURGE ALGUN ERROR HACE LO QUE SE ENCUENTRA EN except:
                    self.elipse = cv2.fitEllipse(self.contornos) # AJUSTA UNA ELIPSE ALREDEDOR DEL CONTORNO. DEVUELVE LA ELIPSE COMO: ((x, y), (MA, ma)) (ubicacion centro, (eje mayor, eje menor))
                    cv2.ellipse(self.drawing, box=self.elipse, color=(0, 255, 0)) 
                    self.area_ellipse = self.elipse[1][0] * self.elipse[1][1] * np.pi # DIBUJA LA ELIPSE
              
                except:
                    pass 
                
            if self.area_ellipse is None: # CARGA EL AREA DE CADA CIRCULO A SU RESPECTIVO FRAME
                self.area_ellipse = 0
            self.parametros.iloc[self.m,2] = (self.area_ellipse)
            self.m = self.m + 1
            
    ##### 6.2.6. FILTRADO #####
            
    ##### SELECCION DE FRAME DONDE SE ENCIENDEN LOS LEDS BLANCOS

            if self.parametros.iloc[self.m-1,0] > 5: # SE UTILIZA EL TIEMPO SETEADO Y LOS FPS A LA HORA DE HACER EL VIDEO PARA EL CALCULO DEL FRAME DONDE INICIA LA CONTRACCIÓN PUPILAR
                self.parametros_luz = self.parametros_luz.append(self.parametros.loc[self.m-1,:],ignore_index=True)
                
    ##### CALCULO DE DERIVADA DEL AREA CON RESPECTO DEL TIEMPO PARA LA CORRÉCTA ELECCIÓN DEL FRAME DONDE SE TERMINA CON CONTRACCION PUPILAR
                
                self.dt= self.tiempo_frames             
                self.da = self.parametros_luz.iloc[:,2].to_numpy() # TRANSFORMA TODOS LOS VALORES QUE SE ENCUENTRAN EN LA COLUMNA "AREA" EN UN ARRAY DE NUMPY
                self.tiempo_grafico = self.parametros_luz.iloc[:,1].to_numpy()
                            
    ##### FILTRO DE MEDIANA AREA#####
                
                try:
                    self.mediana = [self.da[-1],self.da[-2],self.da[-3],self.da[-4],self.da[-5]]
                    self.da[-3] = np.median(self.mediana) 
                      
                except:
                    pass
            
                self.dadt = np.diff(self.da)/self.dt #Calcule la n-ésima diferencia discreta a lo largo del eje dado. dy = valores de Area, dt diferencia entre cada frame
    
    ##### FILTRO DE MEDIANA DERIVADA AREA #####     
                
                try:
                    self.mediana_derivada = [self.dadt[-1],self.dadt[-2],self.dadt[-3],self.dadt[-4],self.dadt[-5]]         
                    self.dadt[-3] = np.median(self.mediana_derivada) 
                
                except:
                    pass
    
                self.tiempo_derivadas = self.parametros_luz.iloc[:,1].to_numpy()
                self.tamano = self.tiempo_derivadas.size -1 # CALCULA LA UBICACION DEL ULTIMO ELEMENTO DEL ARRAY ESTE PROCESO SE HACE DEBIDO QUE PARA TENER UNA DERIVADA NECESITO 2 NUMEROS
                self.tiempo_derivadas_sin_ultimo_elemento = np.delete(self.tiempo_derivadas,self.tamano) # ELIMINA EL ULTIMO ELEMENTO DEL ARRAY
                
    ##### CALCULO DEL FRAME DONDE EMPIEZA LA CONTRACCION (EXTRACCION DE AREA INICIAL Y TIEMPO INICIAL DE PRUEBA) #####
            
                if self.dadt.size >= 7 : # SE VAN A COMPARAR 5 VALORES DE DERIVADAS PERO COMO LAS ULTIMAS 2 ESTAN CALCULADAS A PARTIR DE UN AREA NO FILTRADA SE DESCARTAN
                    self.ultimas_cinco_areas_con_derivada = self.da[-7:-2] #GUARDA LAS ULTIMAS 5 AREAS FILTRADAS           
                    self.derivadas_ultimas_cinco_areas_con_derivadas = self.dadt[-7:-2] # SE GUARDAN LAS DERIVADAS DE LAS ULTIMAS 5 AREAS FILTRADAS CON DERIVADAS
                    self.cantidad_derivadas_menor_cero = np.sum(self.derivadas_ultimas_cinco_areas_con_derivadas < 0) # SE GUARDAN LA CANTIDAD DE DERIVADAS QUE DIERON MENOR A 0. np.sum() suma la canidad de veces que se cumplio la condicion
                    if self.cantidad_derivadas_menor_cero == 5: # SI SE CUMPLE QUE LOS 5 VALORES SON  MENORES A 0 INGRESA A ESTE BUCLE
                        if self.tiempo_inicial is None: # SOLO VA A INGRESAR LA PRIMERA VEZ QUE SE CUMPLA LA CONDICION ANTERIOR UNA VEZ QUE SE GUARDEN LOS VALORES INICIALES
                            self.tiempo_inicial = self.parametros.iloc[self.m-8,1] # COMO SE TRABAJA CON 7 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,3] SE RESTAN 7 
                            self.area_inicial = self.parametros.iloc[self.m-8,2] # COMO SE TRABAJA CON 7 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,2] SE RESTAN 7
                            self.frame_inicial = self.parametros.iloc[self.m-8,0] # COMO SE TRABAJA CON 7 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,0] SE RESTAN 7
                            print("tiempo_inicial:", self.tiempo_inicial)
                            print("area_inicial:", self.area_inicial)
                            print("frame_inicial:", self.frame_inicial)
                            
    ##### CALCULO DEL FRAME DONDE TERMINA LA CONTRACCION (EXTRACCION DE AREA FINAL Y TIEMPO FINAL DE PRUEBA)
                        
                    if self.frame_inicial is not None: #
                    
                        if self.tiempo_final is None:
                         
    ##### CONTROL FRAME FINAL POR MINIMO AREA
                                
                                self.area_minima = np.min(self.da)
                                self.posision_area_minima = (np.argmin(self.da, axis=0) +1 + 5)
                            
                                if self.area_minima < self.area_minima_2:
                                    self.area_minima_2 = self.area_minima
                                    self.posision_area_minima_2 = self.posision_area_minima
                                
                                else:
                                    self.contador = self.contador +1
                                    
                                if self.contador == 5:
                                    self.tiempo_final = self.parametros.iloc[self.posision_area_minima_2-5,1] # COMO SE TRABAJA CON 5 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,3] SE RESTAN 7
                                    self.area_final = self.parametros.iloc[self.posision_area_minima_2-5,2] # COMO SE TRABAJA CON 5 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,3] SE RESTAN 7
                                    self.frame_final = self.parametros.iloc[self.posision_area_minima_2-5,0] # COMO SE TRABAJA CON 5 VALORES HACIA DELANTE EL FRAME INICIAL SE ENCUENTRA 7 LUGARES HACIA ATRAS Y COMO SE COMIENZA DESDE: parametros.iloc[m-1,3] SE RESTAN 7
                                    
                                    print("tiempo_final:",self.tiempo_final)
                                    print("area_final:",self.area_final)
                                    print("frame_final:",self.frame_final)
    
    #### 6.2.7. PARÁMETROS #####
                                    
                                    self.tiempo_respuesta = self.tiempo_final - self.tiempo_inicial
                                    self.area_relativa = (100 - (self.area_final * 100 / self.area_inicial ))
                                    print("TIEMPO PRUEBA:", self.tiempo_respuesta)
                                    print("PORCENTAJE QUE SE ENCOJE PUPILA:",self.area_relativa)
                                    self.resultados = pd.DataFrame(columns=['tiempo_respuesta','area_relativa'],index=range(1))
                                    self.resultados[["tiempo_respuesta"]] = self.tiempo_respuesta
                                    self.resultados[["area_relativa"]] = self.area_relativa
                                    
            else:
                pass
        cv2.destroyAllWindows()
        self.camera.close()
        self.ledIR.close()
        self.led.close()
              
##### INTERFAZ GRÁFICA ######
            
    ##### VENTANA RESULTADOS #####
        
class ventana_resultado(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: ventana_resultado.__init__
        kwds["style"] = kwds.get("style", 0)
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((480, 315))
        self.SetTitle("RESULTADO")

        self.panel_1 = wx.Panel(self, wx.ID_ANY)

        sizer_1 = wx.BoxSizer(wx.VERTICAL)

        grid_sizer_1 = wx.FlexGridSizer(2, 1, 0, 0)
        sizer_1.Add(grid_sizer_1, 1, wx.EXPAND, 0)
        
        ##### DETERMINACION #####
        
        determinacion = None
        try:
            clasificador = joblib.load("modelo.pkl")
            determinacion= clasificador.predict(self.GetParent().hilo1.resultados)
        except:
           pass
        
        if determinacion is None:
            result_text = "Sin Resultados. Repetir Test."
            label_result = wx.StaticText(self.panel_1, wx.ID_ANY, result_text)
        if determinacion is not None:
            if determinacion[0] == "Positivo":
                result_text = "Resultado Positivo \n" + "Tiempo de respuesta: " + str(self.GetParent().hilo1.tiempo_respuesta) + "\nArea Relativa: " + str(self.GetParent().hilo1.area_relativa)        
                label_result = wx.StaticText(self.panel_1, wx.ID_ANY, result_text)
            
            else:
                result_text = "Resultado Negativo \n" + "Tiempo de respuesta: " + str(self.GetParent().hilo1.tiempo_respuesta) + "\nArea Relativa: " + str(self.GetParent().hilo1.area_relativa)        
                label_result = wx.StaticText(self.panel_1, wx.ID_ANY, result_text)

        grid_sizer_1.Add(label_result, 0, wx.ALL | wx.EXPAND, 5)

        self.button_close = wx.Button(self.panel_1, wx.ID_ANY, "Cerrar")
        
        grid_sizer_1.Add(self.button_close, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        grid_sizer_1.AddGrowableRow(0)
        grid_sizer_1.AddGrowableCol(0)

        self.panel_1.SetSizer(sizer_1)

        self.Layout()
        self.Centre()

        self.Bind(wx.EVT_BUTTON, self.onClose, self.button_close)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        
    def onClose(self, event):
        self.GetParent().Show()        
        self.Destroy()


    ##### VENTANA PRINCIPAL #####
            
class ventana_principal(wx.Frame):

    def __init__(self, *args, **kwds):
        wx.Frame.__init__(self, *args, **kwds)
        self.panel_1 = wx.Panel(self, wx.ID_ANY)
        self.label_result = wx.StaticText(self.panel_1, wx.ID_ANY, ("Colocar dispositivo \nPosicionar \nRealizar nuevo test"))
        
    ##### BOTON 1 "POSICIONAR" #####
        
        self.button_posicionar = wx.Button(self.panel_1, wx.ID_ANY, ("Posicionar")) # BOTON 1 
        self.Bind(wx.EVT_BUTTON, self.button_posicionar_evt, self.button_posicionar) # Al hacer click en "comenzar" empieza a funcionar el timer
        self.Bind(wx.EVT_TIMER, self.BOTON1)
        
    ##### BOTON 2 "NUEVO TEST" #####
        
        self.button_nuevo_test = wx.Button(self.panel_1, wx.ID_ANY, ("Nuevo Test")) # BOTON 2 
        self.Bind(wx.EVT_BUTTON, self.button_nuevo_test_evt, self.button_nuevo_test) # Al hacer click en "comenzar" empieza a funcionar el timer
        self.Bind(wx.EVT_TIMER, self.BOTON2)    
        
    ##### BOTON 3 "RESULTADOS" #####
        
        self.button_resultados = wx.Button(self.panel_1, wx.ID_ANY, ("Resultados")) # BOTON 3
        self.__set_properties()
        self.__do_layout()
        self.Bind(wx.EVT_BUTTON, self.button_resultados_evt, self.button_resultados) # Al hacer click en "comenzar" empieza a funcionar el timer
        self.Bind(wx.EVT_TIMER, self.BOTON3)
    
    ##### ACCIONES BOTON 1 #####
   
    def BOTON1(self, event):   
        self.label_result.SetLabel('El sujeto esta posicionado: ') # Mensaje al hacer click EN BOTON 1
        
    ##### ACCIONES BOTON 2 #####
   
    def BOTON2(self, event):   
        self.label_result.SetLabel('Procesando...') # Mensaje al hacer click EN BOTON 2
        
    ##### ACCIONES BOTON 3 #####
   
    def BOTON3(self, event):     
        self.label_result.SetLabel('Los resultados son los siguientes: ') # Mensaje al hacer click EN BOTON 3
        
    ##### TAMAÑO DE LA VENTANA #####
        
    def __set_properties(self):
        self.SetTitle(("PUPILÓGRAFO"))
        self.SetSize((480, 315))
        self.label_result.SetFont(wx.Font(15, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "")) #Caracterizticas de la fuente de los mensajes
        self.button_posicionar.SetFont(wx.Font(19, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "")) #Caracterizticas de la fuente del botón 
        self.button_nuevo_test.SetFont(wx.Font(19, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "")) #Caracterizticas de la fuente del botón
        self.button_resultados.SetFont(wx.Font(19, wx.DEFAULT, wx.NORMAL, wx.NORMAL, 0, "")) #Caracterizticas de la fuente del botón
        self.button_resultados.Hide()
        
    ##### DISEÑO DE LA VENTANA #####
    
    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_1 = wx.GridSizer(4, 1, 0, 0)
        grid_sizer_1.Add(self.label_result, 0, wx.ALL, 5)
        grid_sizer_1.Add(self.button_posicionar, 0, wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT | wx.ALL, 5)
        grid_sizer_1.Add(self.button_nuevo_test, 0, wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT | wx.ALL, 5)
        grid_sizer_1.Add(self.button_resultados, 0, wx.ALIGN_BOTTOM | wx.ALIGN_RIGHT | wx.ALL, 5)
        sizer_2.Add(grid_sizer_1, 1, wx.EXPAND, 0)
        self.panel_1.SetSizer(sizer_2)
        sizer_1.Add(self.panel_1, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        self.Layout()
        
    ##### ACCIONES AL HACER CLICK EN "POSICIONAR" #####
        
    def button_posicionar_evt(self, event):
        
        ##### INICIALIZA LA CAMARA Y CONFIGURA #####
            
        self.camera = PiCamera()
        self.camera.resolution = (1280, 720) # SE CONFIGURA LA RESOLUCIóN EN 720P
        self.camera.framerate = 60 # SE CONFIGURA LA LOS FRAMES EN 60
        time.sleep(0.1) # CALENTAMIENTO CÁMARA
        tiempo_inicio = time.time() # SE GUARDA EL TIEMPO INICIAL 
        tiempo_final = 0 # SE INSTANCIA EL TIEMPO FINAL
        
        ##### L.E.D.s #####
        
        self.led = LED(6) #LED BLANCO
        self.ledIR = LED(16) #LED INFRAROJO
        self.ledIR.on() #ENCIENDE EL LED INFRAROJO
        
        ##### CAPTURA Y VISUALIZACIÓN DE STREAM #####
        
        captura_posicionamiento = PiRGBArray(self.camera, size=(1280,720))
        for frame in self.camera.capture_continuous(captura_posicionamiento, format="bgr", use_video_port=True):
            tiempo_final = time.time() - tiempo_inicio
            if tiempo_final < 10: # SE CONFIGURA PARA QUE MUESTRE UN STREAM DE 10 SEGUNDOS
                image= frame.array
                imageOut = cv2.resize(image,(480,320), interpolation=cv2.INTER_CUBIC) # SE REALIZA UN REAJUSTE DE LA IMAGEN PARA SU CORRECTA VISUALIZACIÓN EN EL DISPLAY
                cv2.circle(imageOut, (244,102),12, (0,0,0), -1) # SE DIBUJA UN CIRCULO NEGRO EN EL CENTRO DEL BRILLO DE LOS L.E.D.s
                
                ##### CONFIGURACIÓN PARA LA CORRECTA VISUALIZACIÓN EN DISPLAY #####
                
                cv2.namedWindow("Posicionamiento", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Posicionamiento", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Posicionamiento",imageOut)
                
                captura_posicionamiento.truncate(0)
                key = cv2.waitKey(1) 
            else:
               break
        cv2.destroyAllWindows()
        
        ##### CIERRA LA CAMARA Y SALIDAS DE LOS L.E.D.s #####

        self.camera.close()
        self.ledIR.close()
        self.led.close()
        event.Skip()
        
    ##### ACCIONES AL HACER CLICK EN "NUEVO TEST" #####
        
    def button_nuevo_test_evt(self, event):
        self.hilo1 = hilovalores(parent=self) # INSTANCIA EL HILO
        self.hilo1.start() # INICIA EL HILO
        self.hilo1.join() # ESPERA QUE EL HILO TERMINE PARA CONTINUAR
        self.ventana = ventana_resultado(self, style=wx.DEFAULT_FRAME_STYLE)#wx.MAXIMIZE
        self.Hide()
        self.ventana.Show()
        event.Skip()

    ##### ACCIONES AL HACER CLICK EN "RESULTADOS" #####       
        
    def button_resultados_evt(self, event):
        event.Skip()

##### ABRE LA VENTANA AL INICIAR EL PROGRAMA #####
        
class MyApp(wx.App):
    
    def OnInit(self):
        frame_1 = ventana_principal(None, wx.ID_ANY, "", style=wx.DEFAULT_FRAME_STYLE)#wx.MAXIMIZE
        self.SetTopWindow(frame_1)
        frame_1.Show()
        return True

if __name__ == "__main__":
    gettext.install("app") 
    app = MyApp(0)
    app.MainLoop()