% Cargar la base de datos
clear; clc;
a=load('data.txt');
muestrasPositivas = 416;

% Dividir el número de muestras en negativos y positivos

p=a(1:muestrasPositivas,1:10);
n=a(muestrasPositivas + 1:end,1:10);

% Seleccionar la cantidad de datos para entrenamiento
numP = 278;
numN = 125;

% Obtener las matrices de entrenamiento, salidas deseadas y prueba
entrenamiento=[p(1:numP,:);n(1:numN,:)];
deseada=[ones(numP,1);ones(numN,1).*-1];
prueba=[p(numP + 1:end,:);n(numN + 1:end,:)];

% Obtener el modelo de SVM utilizando RBF
SVMModel = fitcsvm(entrenamiento,deseada,'Standardize',true,...
        'KernelFunction','rbf');
[RBF_Etiquetas,RBF_Resultado] = predict(SVMModel,prueba); 

% Seleccionar el número de positivos y negativos en el módelo SVM
[sP,ss] = size(prueba);
positivos = sP - 30;

% Obtener el número de clasificaciones positivas del SVM
RBF_VP=0;
for i=1:positivos
if (RBF_Etiquetas(i,1)==1)
    RBF_VP=RBF_VP+1;
end
end

% Calcular la sensibilidad del SVM con RBF
RBF_R_Sensibilidad=(RBF_VP/positivos)*100;

%Obtener el número de clasificaciones negativas del SVM
RBF_VN=0;
for i=positivos + 1:sP%%17
if (RBF_Etiquetas(i,1)==-1)
    RBF_VN=RBF_VN+1;
end
end

%Calcular la especificidad del SVM con RBF
RBF_R_Especificidad=(RBF_VN/(sP-positivos))*100;

%Obtener el número de verdaderos positivos y negativos y de falsos
%positivos y negativos
RBF_VP;
RBF_FN=positivos-RBF_VP;
EjemplosPositivos_RBF=RBF_VP+RBF_FN;
RBF_VN;
RBF_FP=(prueba - positivos)-RBF_VN;
EjemplosNegativos_RBF=RBF_VN+RBF_FP;

%Obtener el módelo de SVM con función Polinomial
SVMMode2 = fitcsvm(entrenamiento,deseada,'Standardize',true,...
        'KernelFunction','polynomial', 'PolynomialOrder' ,6,'BoxConstraint',1);
[POLY_Etiquetas,POLY_Resultado] = predict(SVMMode2,prueba);


POLY_VP=0;
for i=1:positivos
if (POLY_Etiquetas(i,1)==1)
    POLY_VP=POLY_VP+1;
end
end
POLY_R_Sensibilidad=(POLY_VP/216)*100;
POLY_VN=0;
for i=positivos + 1:sP%%17
if (POLY_Etiquetas(i,1)==-1)
    POLY_VN=POLY_VN+1;
end
end
POLY_R_Especificidad=(POLY_VN/(sP - positivos))*100;

POLY_VP ;
POLY_FN=positivos-POLY_VP;
EjemplosPositivos_POLY=POLY_VP+POLY_FN;
POLY_VN;
POLY_FP=(prueba - positivos)-POLY_VN;
EjemplosNegativos_POLY=POLY_VN+POLY_FP;
