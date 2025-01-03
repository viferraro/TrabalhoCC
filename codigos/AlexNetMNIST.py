# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime
# from carbontracker.tracker import CarbonTracker
# from carbontracker import parser
# from thop import profile
# from torchsummary import summary
# import pynvml
# import seaborn as sns
# import matplotlib.pyplot as plt
# import io
# import sys
#
# # Valor usado para inicializar o gerador de números aleatórios
# SEED = 10
#
# # Verificar se a GPU está disponível
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# for i in range(1):
#     # Inicialização do NVML para monitoramento da GPU
#     pynvml.nvmlInit()
#
#
#     # Defina uma função para criar um diretório com incremento, se ele já existir
#     def create_incremented_dir(base_dir, subfolder_name):
#         i = 1
#         parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
#         while os.path.exists(parent_dir):
#             i += 1
#             parent_dir = os.path.join(base_dir, f"{subfolder_name}_{i}")
#         os.makedirs(parent_dir)
#         return parent_dir
#
#
#     # Crie o diretório pai 'alexNetMNIST_' com incremento se necessário
#     parent_dir = create_incremented_dir('resultados7', 'alexNetMNIST')
#     print(f'Diretório criado: {parent_dir}')
#
#     # Crie o diretório 'AlexNetCarbon'
#
#     carbon_dir = create_incremented_dir(parent_dir, 'alexNet_carbon')
#     print(f'Diretório Carbon criado: {carbon_dir}')
#
#     # Definições iniciais
#     max_epochs = 20
#     train_times = []
#     train_powers = []
#     tracker = CarbonTracker(epochs=max_epochs, monitor_epochs=-1, interpretable=True,
#                             log_dir=f"./{carbon_dir}/",
#                             log_file_prefix="cbt")
#
#     log_dir = f"./{carbon_dir}/"
#     all_logs = os.listdir(log_dir)
#     std_logs = [f for f in all_logs if f.endswith('_carbontracker.log')]
#     missing_logs = ['epoch_{}_carbontracker.log'.format(i) for i in range(max_epochs) if
#                     'epoch_{}_carbontracker.log'.format(i) not in all_logs]
#     for f in missing_logs:
#         log_file = f + "_carbontracker.log"
#         if log_file in std_logs:
#             std_logs.remove(log_file)
#
#     # Agora você pode chamar as funções do parser com segurança
#     parser.print_aggregate(log_dir)
#     logs = parser.parse_all_logs(log_dir)
#     first_log = logs[0]
#
#     print(f"Output file name: {first_log['output_filename']}")
#     print(f"Standard file name: {first_log['standard_filename']}")
#     print(f"Stopped early: {first_log['early_stop']}")
#     print(f"Measured consumption: {first_log['actual']}")
#     print(f"Predicted consumption: {first_log['pred']}")
#     print(f"Measured GPU devices: {first_log['components']['gpu']['devices']}")
#
#     # Carregar e normalizar o MNIST
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#
#     full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
#
#     # Dividir o conjunto de treino em treino e validação
#     train_size = int(0.8 * len(full_train_dataset))
#     val_size = len(full_train_dataset) - train_size
#     train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#     # Definição da arquitetura da rede neural AlexNet
#     class AlexNet(nn.Module):
#         def __init__(self, num_classes=10):
#             super(AlexNet, self).__init__()
#             self.features = nn.Sequential(
#                 nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2),
#                 nn.Conv2d(64, 192, kernel_size=3, padding=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2),
#                 nn.Conv2d(192, 384, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(384, 256, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2),
#             )
#             self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
#             self.classifier = nn.Sequential(
#                 nn.Dropout(),
#                 nn.Linear(256 * 2 * 2, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(),
#                 nn.Linear(4096, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(4096, num_classes),
#             )
#
#         def forward(self, x):
#             x = self.features(x)
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.classifier(x)
#             return x
#
#     model = AlexNet().to(device)
#     print(model)
#
#     # Salvar a saída padrão original
#     original_stdout = sys.stdout
#
#     # Redirecionar a saída padrão para um buffer de string
#     sys.stdout = buffer = io.StringIO()
#
#     # Chamar a função summary
#     summary(model, (1, 28, 28))
#
#     # Obter o valor da string do buffer
#     summary_str = buffer.getvalue()
#
#     # Restaurar a saída padrão original
#     sys.stdout = original_stdout
#
#     # Salvar a string de resumo em um arquivo
#     with open(f'{parent_dir}/model_summary.txt', 'w') as f:
#         f.write(summary_str)
#
#     # Salvar a saída padrão original novamente
#     original_stdout = sys.stdout
#
#     # Redirecionar a saída padrão para um arquivo
#     sys.stdout = open(f'{parent_dir}/output.txt', 'w')
#
#
#     # Função para treinar e validar um modelo
#     def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs):
#         model.train()
#         start_time = datetime.now()
#         tracker.epoch_start()
#         for epoch in range(epochs):
#             tracker.epoch_start()
#             running_loss = 0.0
#             correct = 0
#             total = 0
#             for i, data in enumerate(train_loader, 0):
#                 inputs, labels = data[0].to(device), data[1].to(device)
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 # Medir o consumo de energia
#                 handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#                 info = pynvml.nvmlDeviceGetPowerUsage(handle)
#                 power_usage = info / 1000.0
#                 train_powers.append(power_usage)
#             train_loss = running_loss / len(train_loader)
#             train_accuracy = correct / total
#             print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
#
#             # Validação
#             model.eval()
#             val_loss = 0.0
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for data in val_loader:
#                     images, labels = data[0].to(device), data[1].to(device)
#                     outputs = model(images)
#                     loss = criterion(outputs, labels)
#                     val_loss += loss.item()
#                     _, predicted = torch.max(outputs.data, 1)
#                     total += labels.size(0)
#                     correct += (predicted == labels).sum().item()
#             tracker.epoch_end()
#             val_loss /= len(val_loader)
#             val_accuracy = correct / total
#             print(f'Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
#         end_time = datetime.now()
#         train_time = (end_time - start_time)
#         train_times.append(train_time.total_seconds())
#         tracker.epoch_end()
#         return train_loss, train_accuracy, val_loss, val_accuracy, train_time, power_usage
#
#
#     # Treinamento e seleção do melhor modelo entre 10 candidatos
#     num_models = 10
#     avg_valid_loss = []
#     best_model_idx = -1
#     best_model = model
#     models = []
#     metrics = []
#     avg_metrics = []
#     for i in range(num_models):
#         print("______________________________________________________________________________________________________")
#         print(f'Training model {i + 1}/{num_models}')
#         input = torch.randn(1, 1, 28, 28).to(device)
#         model = AlexNet().to(device)
#         flops, params = profile(model, inputs=(input,), verbose=False)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
#         train_loss, train_accuracy, val_loss, val_accuracy, train_time, power_usage = (
#             train_and_validate(model, train_loader, val_loader, criterion, optimizer, 20))
#         metrics.append((train_loss, train_accuracy, val_loss, val_accuracy, train_time.total_seconds(), power_usage))
#         # Calcular a média das métricas após o treino de cada modelo
#         avg_train_loss = np.mean([m[0] for m in metrics])
#         avg_train_accuracy = np.mean([m[1] for m in metrics])
#         avg_val_loss = np.mean([m[2] for m in metrics])
#         avg_val_accuracy = np.mean([m[3] for m in metrics])
#         print(f'Model {i + 1}: Avg Train Loss: {avg_train_loss:.4f}, Avg Train Accuracy: {avg_train_accuracy:.4f}, '
#               f'Avg Val Loss: {avg_val_loss:.4f}, Avg Val Accuracy: {avg_val_accuracy:.4f}')
#         print(f'Tempo de treino: {train_time}')
#         print(f'FLOPs: {flops}')
#         print(f'Parâmetros: {params}')
#         print(f'Power usage: {power_usage} W')
#         avg_valid_loss.append(avg_val_loss)
#         avg_metrics.append(
#             (avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy, train_time.total_seconds(),
#              power_usage))
#         models.append(model)
#
#     # Crie um DataFrame com as métricas médias e salve-o em um arquivo Excel
#     df_metrics = pd.DataFrame(avg_metrics, columns=['avg_Train Loss', 'avg_Train Accuracy', 'avg_Val Loss',
#                                                     'avg_Val Accuracy', 'TrainTime', 'PowerUsage'])
#
#     # Adiciona uma coluna 'Modelo_x' ao DataFrame
#     modelos = ['Modelo_' + str(i + 1) for i in range(num_models)]
#     df_metrics.insert(0, 'Model', modelos)
#
#     # Salva as métricas de todos os modelos em um único arquivo no diretório pai 'leNet_x'
#     df_metrics.to_excel(f'{parent_dir}/models_metrics.xlsx', index=False)
#
#     # Seleciona o melhor modelo com base na menor perda de validação.
#     best_model_index = avg_valid_loss.index(min(avg_valid_loss))
#     best_model = models[best_model_index]
#     print('************************************************************************************************')
#     print(
#         f'O melhor modelo é o Modelo {best_model_index + 1} com a menor perda média de validação: {min(avg_valid_loss):.4f}')
#
#     print('************************************************************************************************')
#     # Calcular a média dos tempos de treino e power usage
#     avg_train_time = np.mean(train_times)
#     avg_power_usage = np.mean(train_powers)
#     # avg_metrics.append((avg_train_time, avg_power_usage))
#     print(f'Average Train Time: {avg_train_time} seconds')
#     print(f'Average Power Usage: {avg_power_usage} W')
#
#     # Inicializa listas para armazenar métricas de todas as inferências
#     accuracies = []
#     precisions = []
#     recalls = []
#     f1_scores = []
#     test_times = []
#
#     # Realiza 10 inferências e armazena as métricas
#     for i in range(10):
#         y_true = []
#         y_pred = []
#         start_time_test = datetime.now()
#         best_model.eval()
#         with torch.no_grad():
#             for data in test_loader:
#                 images, labels = data[0].to(device), data[1].to(device)
#                 outputs = best_model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 y_true.extend(labels.cpu().numpy())
#                 y_pred.extend(predicted.cpu().numpy())
#         end_time_test = datetime.now()
#
#         # Calcula as métricas para a inferência atual
#         accuracies.append(accuracy_score(y_true, y_pred))
#         precisions.append(precision_score(y_true, y_pred, average='macro'))
#         recalls.append(recall_score(y_true, y_pred, average='macro'))
#         f1_scores.append(f1_score(y_true, y_pred, average='macro'))
#         test_times.append((end_time_test - start_time_test).total_seconds())
#
#     # Calcula a média das métricas
#     mean_accuracy = sum(accuracies) / len(accuracies)
#     mean_precision = sum(precisions) / len(precisions)
#     mean_recall = sum(recalls) / len(recalls)
#     mean_f1 = sum(f1_scores) / len(f1_scores)
#     mean_test_time = sum(test_times) / len(test_times)
#
#     # Imprime as médias das métricas
#     print(f'Média da Acurácia: {mean_accuracy}\n')
#     print(f'Média da Precisão: {mean_precision}\n')
#     print(f'Média do Recall: {mean_recall}\n')
#     print(f'Média do F1 Score: {mean_f1}\n')
#     print(f'Média do Tempo de Teste: {mean_test_time} segundos\n')
#
#     sys.stdout.close()
#     sys.stdout = original_stdout
#
#     # Calcular a matriz de confusão
#     cm = confusion_matrix(y_true, y_pred)
#
#     # Plotar a matriz de confusão
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
#     plt.xlabel('Predicted')
#     plt.ylabel('Truth')
#
#     # Salvar a figura
#     plt.savefig(f'{parent_dir}/confusion_matrix.png')
#     plt.close()
#
#     # #  Salvar as métricas do melhor modelo no diretório pai 'leNet_'
#     # with open(f'{parent_dir}/best_model_metrics.txt', 'w') as f:
#     #     f.write(f'Accuracy: {accuracy}\n')
#     #     f.write(f'Precision: {precision}\n')
#     #     f.write(f'Recall: {recall}\n')
#     #     f.write(f'F1 Score: {f1}\n')
#     #     f.write(f'Test Time: {test_time}\n')
#     #     f.write(f'Seconds: {test_time.total_seconds()}\n')
#
#     # Salva as médias das métricas em um arquivo
#     with open(f'{parent_dir}/average_model_metrics.txt', 'w') as f:
#         f.write(f'Média da Acurácia: {mean_accuracy}\n')
#         f.write(f'Média da Precisão: {mean_precision}\n')
#         f.write(f'Média do Recall: {mean_recall}\n')
#         f.write(f'Média do F1 Score: {mean_f1}\n')
#         f.write(f'Média do Tempo de Teste: {mean_test_time} segundos\n')
#
#     pynvml.nvmlShutdown()
#     tracker.stop()
#     print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')
#
#     print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from thop import profile
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from carbontracker.tracker import CarbonTracker
from carbontracker import parser
import pynvml

# Constante para inicialização do gerador de números aleatórios
SEED = 10

# Verifica se a GPU está disponível
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dispositivo)

# Inicializa o NVML para monitoramento da GPU
pynvml.nvmlInit()

# Função para criar um diretório com incremento, se necessário
def criar_diretorio_incrementado(diretorio_base, nome_subpasta):
    contador = 1
    diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    while os.path.exists(diretorio_pai):
        contador += 1
        diretorio_pai = os.path.join(diretorio_base, f"{nome_subpasta}_{contador}")
    os.makedirs(diretorio_pai)
    return diretorio_pai

# Cria o diretório pai 'alexNetMNIST_' com incremento, se necessário
diretorio_pai = criar_diretorio_incrementado('resultadosAlexNet', 'alexNetMNIST')
print(f'Diretório criado: {diretorio_pai}')

# Cria o diretório 'AlexNetCarbon'
diretorio_carbon = criar_diretorio_incrementado(diretorio_pai, 'alexNet_carbono')
print(f'Diretório Carbono criado: {diretorio_carbon}')

# Definições iniciais
maximo_epocas = 20
tempos_treino = []
potencias_treino = []
tracker = CarbonTracker(epochs=maximo_epocas, monitor_epochs=-1, interpretable=True,
                           log_dir=f"./{diretorio_carbon}/",
                           log_file_prefix="cbt")

# Carrega e normaliza o MNIST
transformacao = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

conjunto_treino_completo = datasets.MNIST(root='./data', train=True, download=True, transform=transformacao)
conjunto_teste = datasets.MNIST(root='./data', train=False, download=True, transform=transformacao)

# Divide o conjunto de treino em treino e validação
tamanho_treino = int(0.8 * len(conjunto_treino_completo))
tamanho_validacao = len(conjunto_treino_completo) - tamanho_treino
conjunto_treino, conjunto_validacao = random_split(conjunto_treino_completo, [tamanho_treino, tamanho_validacao])

carregador_treino = DataLoader(conjunto_treino, batch_size=64, shuffle=True)
carregador_validacao = DataLoader(conjunto_validacao, batch_size=64, shuffle=False)
carregador_teste = DataLoader(conjunto_teste, batch_size=64, shuffle=False)

# Definição da arquitetura da rede neural AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classificador = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classificador(x)
        return x

modelo = AlexNet().to(dispositivo)
print(modelo)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um buffer de string
sys.stdout = buffer = io.StringIO()

# Chamar a função summary
summary(modelo, (1, 28, 28))

# Obter o valor da string do buffer
resumo_str = buffer.getvalue()

# Restaurar a saída padrão original
sys.stdout = saida_padrao_original

# Salvar a string de resumo em um arquivo
with open(f'{diretorio_pai}/resumo_modelo.txt', 'w') as f:
    f.write(resumo_str)

# Salvar a saída padrão original
saida_padrao_original = sys.stdout

# Redirecionar a saída padrão para um arquivo
sys.stdout = open(f'{diretorio_pai}/saida.txt', 'w')

# Função para treinar e validar um modelo
def treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, epocas):
    modelo.train()
    inicio_tempo = datetime.now()
    tracker.epoch_start()
    for epoca in range(epocas):
        tracker.epoch_start()
        perda_acumulada = 0.0
        corretos = 0
        total = 0
        for i, dados in enumerate(carregador_treino, 0):
            entradas, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            otimizador.zero_grad()
            saidas = modelo(entradas)
            perda = criterio(saidas, rotulos)
            perda.backward()
            otimizador.step()
            perda_acumulada += perda.item()
            _, previstos = torch.max(saidas.data, 1)
            total += rotulos.size(0)
            corretos += (previstos == rotulos).sum().item()
            # Medir o consumo de energia
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetPowerUsage(handle)
            consumo_energia = info / 1000.0
            potencias_treino.append(consumo_energia)
        perda_treino = perda_acumulada / len(carregador_treino)
        acuracia_treino = corretos / total
        print(f'Época {epoca + 1}, Perda Treino: {perda_treino:.4f}, Precisão Treino: {acuracia_treino:.4f}')

        # Validação
        modelo.eval()
        perda_validacao = 0.0
        corretos = 0
        total = 0
        with torch.no_grad():
            for dados in carregador_validacao:
                imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)
                perda_validacao += perda.item()
                _, previstos = torch.max(saidas.data, 1)
                total += rotulos.size(0)
                corretos += (previstos == rotulos).sum().item()
        tracker.epoch_end()
        perda_validacao /= len(carregador_validacao)
        acuracia_validacao = corretos / total
        print(f'Época {epoca + 1}, Perda Validação: {perda_validacao:.4f}, Precisão Validação: {acuracia_validacao:.4f}')
    fim_tempo = datetime.now()
    tempo_treino = (fim_tempo - inicio_tempo)
    tempos_treino.append(tempo_treino.total_seconds())
    tracker.epoch_end()
    return perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia

# Treinamento e seleção do melhor modelo entre 10 candidatos
numero_modelos = 10
medias_perda_validacao = []
indice_melhor_modelo = -1
melhor_modelo = modelo
modelos = []
metricas = []
media_metricas = []
for i in range(numero_modelos):
    print("______________________________________________________________________________________________________")
    print(f'Treinando modelo {i + 1}/{numero_modelos}')
    entrada = torch.randn(1, 1, 28, 28).to(dispositivo)
    modelo = AlexNet().to(dispositivo)
    flops, parametros = profile(modelo, inputs=(entrada,), verbose=False)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=1e-4)
    perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino, consumo_energia = (
        treinar_e_validar(modelo, carregador_treino, carregador_validacao, criterio, otimizador, 20))
    metricas.append((perda_treino, acuracia_treino, perda_validacao, acuracia_validacao, tempo_treino.total_seconds(), consumo_energia))
    # Calcular a média das métricas após o treino de cada modelo
    media_perda_treino = np.mean([m[0] for m in metricas])
    media_acuracia_treino = np.mean([m[1] for m in metricas])
    media_perda_validacao = np.mean([m[2] for m in metricas])
    media_acuracia_validacao = np.mean([m[3] for m in metricas])
    print(f'Modelo {i + 1}: Média Perda Treino: {media_perda_treino:.4f}, Média Acurácia Treino: {media_acuracia_treino:.4f}, '
          f'Média Perda Validação: {media_perda_validacao:.4f}, Média Acurácia Validação: {media_acuracia_validacao:.4f}')
    print(f'Tempo de treino: {tempo_treino}')
    print(f'FLOPs: {flops}')
    print(f'Parâmetros: {parametros}')
    print(f'Consumo de energia: {consumo_energia} W')
    medias_perda_validacao.append(media_perda_validacao)
    media_metricas.append(
        (media_perda_treino, media_acuracia_treino, media_perda_validacao, media_acuracia_validacao, tempo_treino.total_seconds(),
         consumo_energia))
    modelos.append(modelo)

# Cria um DataFrame com as métricas médias e salva em um arquivo Excel
df_metricas = pd.DataFrame(media_metricas, columns=['Média Perda Treino', 'Média Precisão Treino', 'Média Perda Validação',
                                                    'Média Precisão Validação', 'TempoTreino', 'ConsumoEnergia'])

# Adiciona uma coluna 'Modelo_x' ao DataFrame
nomes_modelos = ['Modelo_' + str(i + 1) for i in range(numero_modelos)]
df_metricas.insert(0, 'Modelo', nomes_modelos)

# Salva as métricas de todos os modelos em um único arquivo no diretório pai
df_metricas.to_excel(f'{diretorio_pai}/metricas_modelos.xlsx', index=False)

# Seleciona o melhor modelo com base na menor perda de validação
indice_melhor_modelo = medias_perda_validacao.index(min(medias_perda_validacao))

melhor_modelo = modelos[indice_melhor_modelo]
print('************************************************************************************************')
print(f'O melhor modelo é o {nomes_modelos[indice_melhor_modelo]} com a menor média de perda de validação: {media_perda_validacao:.4f}')
print('************************************************************************************************')


# Calcular a média dos tempos de treino e consumo de energia
media_tempo_treino = np.mean(tempos_treino)
media_consumo_energia = np.mean(potencias_treino)
print(f'Tempo Médio de Treino: {media_tempo_treino} segundos')
print(f'Consumo Médio de Energia: {media_consumo_energia} W')

# Inicializa listas para armazenar métricas de todas as inferências
acuracias = []
precisoes = []
revocacoes = []
pontuacoes_f1 = []
tempos_teste = []

# Realiza 10 inferências e armazena as métricas
for i in range(10):
    y_verdadeiros = []
    y_previstos = []
    inicio_tempo_teste = datetime.now()
    melhor_modelo.eval()
    with torch.no_grad():
        for dados in carregador_teste:
            imagens, rotulos = dados[0].to(dispositivo), dados[1].to(dispositivo)
            saidas = melhor_modelo(imagens)
            _, previstos = torch.max(saidas.data, 1)
            y_verdadeiros.extend(rotulos.cpu().numpy())
            y_previstos.extend(previstos.cpu().numpy())
    fim_tempo_teste = datetime.now()

    # Calcula as métricas para a inferência atual
    acuracias.append(accuracy_score(y_verdadeiros, y_previstos))
    precisoes.append(precision_score(y_verdadeiros, y_previstos, average='macro'))
    revocacoes.append(recall_score(y_verdadeiros, y_previstos, average='macro'))
    pontuacoes_f1.append(f1_score(y_verdadeiros, y_previstos, average='macro'))
    tempos_teste.append((fim_tempo_teste - inicio_tempo_teste).total_seconds())

# Calcula a média das métricas
media_acuracia = sum(acuracias) / len(acuracias)
media_precisao = sum(precisoes) / len(precisoes)
media_revocacao = sum(revocacoes) / len(revocacoes)
media_f1 = sum(pontuacoes_f1) / len(pontuacoes_f1)
media_tempo_teste = sum(tempos_teste) / len(tempos_teste)

# Imprime as médias das métricas
print(f'Média da Acurácia: {media_acuracia}')
print(f'Média da Precisão: {media_precisao}')
print(f'Média do Recall: {media_revocacao}')
print(f'Média do F1 Score: {media_f1}')
print(f'Média do Tempo de Teste: {media_tempo_teste} segundos')

sys.stdout.close()
sys.stdout = saida_padrao_original

# Calcular a matriz de confusão
cm = confusion_matrix(y_verdadeiros, y_previstos)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.xlabel('Previstos')
plt.ylabel('Verdadeiros')

# Salvar a figura
plt.savefig(f'{diretorio_pai}/matriz_confusao.png')
plt.close()

# Salva as médias das métricas em um arquivo
with open(f'{diretorio_pai}/metricas_medias_modelo.txt', 'w') as f:
    f.write(f'Média da Acurácia: {media_acuracia}\n')
    f.write(f'Média da Precisão: {media_precisao}\n')
    f.write(f'Média do Recall: {media_revocacao}\n')
    f.write(f'Média do F1 Score: {media_f1}\n')
    f.write(f'Média do Tempo de Teste: {media_tempo_teste} segundos\n')

pynvml.nvmlShutdown()
tracker.stop()
print('Treinamento concluído. Os resultados foram salvos nos arquivos especificados.')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')