#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace opt = boost::program_options;

#include "cublas_v2.h"

// собственно возвращает значение линейной интерполяции
double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    // делаем значение y(щначение клетки)используя формулу линейной интерполяции
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}




void initMatrix(std::unique_ptr<double[]> &arr ,int N){
        
          arr[0] = 10.0;
          arr[N-1] = 20.0;
          arr[(N-1)*N + (N-1)] = 30.0;
          arr[(N-1)*N] = 20.0;
              // инициализируем и потом сразу отправим на девайс
        for (size_t i = 1; i < N-1; i++)
        {
            arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
            arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
            arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
            arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
        }
}




void saveMatrixToFile(const double* matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Устанавливаем ширину вывода для каждого элемента
    int fieldWidth = 10; // Ширина поля вывода, можно настроить по вашему усмотрению

    // Записываем матрицу в файл с выравниванием столбцов
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
}


int main(int argc, char const *argv[])
{
    // парсим аргументы
    opt::options_description desc("опции");
    desc.add_options()
        ("accuracy",opt::value<double>()->default_value(1e-6),"точность")
        ("cellsCount",opt::value<int>()->default_value(256),"размер матрицы")
        ("iterCount",opt::value<int>()->default_value(1000000),"количество операций")
        ("help","помощь")
    ;

    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    
    // и это всё было только ради того чтобы спарсить аргументы.......

    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed\n";
        std::cout << "NAME: " << cublasGetStatusName(stat) << '\n';
        std::cout << "DES: " << cublasGetStatusString(stat) << '\n';
        return 3;
        }

    double error = 1.0;
    int iter = 0;

    std::unique_ptr<double[]> A(new double[N*N]);
    std::unique_ptr<double[]> Anew(new double[N*N]);
    

    double alpha = -1.0;
    
    int idx=0;

    // инициализация вспомогательных переменных cublas


    initMatrix(std::ref(A),N);
    initMatrix(std::ref(Anew),N);
   
    auto start = std::chrono::high_resolution_clock::now();
    double* curmatrix = A.get();
    double* prevmatrix = Anew.get();
    
    
    #pragma acc data copyin(idx,prevmatrix[0:N*N],curmatrix[0:N*N],N,alpha)
    {
    while (iter < countIter && iter<10000000 && error > accuracy){
             // провереное эксперементальным путём оптимальное количество банд и размер вектора для расчёта матрицы 1024^2
            #pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(1024) present(curmatrix,prevmatrix)
            for (size_t i = 1; i < N-1; i++)
            {
                for (size_t j = 1; j < N-1; j++)
                {
                    curmatrix[i*N+j]  = 0.25 * (prevmatrix[i*N+j+1] + prevmatrix[i*N+j-1] + prevmatrix[(i-1)*N+j] + prevmatrix[(i+1)*N+j]);
                }
            }

    
            if ((iter+1)%10000 == 0){
                #pragma acc data present(prevmatrix,curmatrix) wait
                #pragma acc host_data use_device(curmatrix,prevmatrix)
                {
                    // разница двух матриц, результат записываем в матрицу предыдущей итерации
                    stat = cublasDaxpy(handle,N*N,&alpha,curmatrix,1,prevmatrix,1);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cout << "cublasDaxpy failed\n";
                        std::cout << "NAME: " << cublasGetStatusName(stat) << '\n';
                        std::cout << "DES: " << cublasGetStatusString(stat) << '\n';
                        }

                    // индекс максимального по модулю элемента
                    stat = cublasIdamax(handle,N*N,prevmatrix,1,&idx);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cout << "cublasIdamax failed\n";
                        std::cout << "NAME: " << cublasGetStatusName(stat) << '\n';
                        std::cout << "DES: " << cublasGetStatusString(stat) << '\n';
                        }
                    

                }
                #pragma acc update self(prevmatrix[idx-1])
                    error = fabs(prevmatrix[idx-1]);
                    std::cout << "iteration: "<<iter+1 << ' ' <<"error: "<<error << std::endl;
                #pragma acc host_data use_device(curmatrix,prevmatrix)
                {
                    stat = cublasDcopy(handle,N*N,curmatrix,1,prevmatrix,1);
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cout << "cublasDcopy failed\n";
                        std::cout << "NAME: " << cublasGetStatusName(stat) << '\n';
                        std::cout << "DES: " << cublasGetStatusString(stat) << '\n';
                        }
                }
                
            }
                        
            double* temp = prevmatrix;
            prevmatrix = curmatrix;
            curmatrix = temp;
        iter++;

    }
    
    cublasDestroy(handle);
    #pragma acc update self(curmatrix[0:N*N])
    }

    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
    
    std::cout<<"time: " << time_s<<" error: "<<error << " iterarion: " << iter<<std::endl;
    
    if (N <=13){
        
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                /* code */
                std::cout << A[i*N+j] << ' ';
                
            }
            std::cout << std::endl;
        }
    }
    saveMatrixToFile(std::ref(curmatrix), N , "matrix.txt");
    A = nullptr;
    Anew = nullptr;

    
    

    return 0;
}
