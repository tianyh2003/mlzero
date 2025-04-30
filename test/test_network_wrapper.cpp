
#include "../network_wrapper.hpp"
#include "../common.h"
#include "../logger.hpp"

#include <cstdlib>

static void predict_operation(NetworkWrapper* nw, int batch_size, float *input, float *pi, float *v) {
    nw->predict_single(batch_size, input, pi, v);
}
static void predict_operation_test(NetworkWrapper* nw, int batch_size, float *input, float *pi, float *v) {
    nw->predict(batch_size, input, pi, v);
}
static void predict_operation_trt(NetworkWrapper* nw, int batch_size, float *input, float *pi, float *v) {
    nw->predict_only_buffer(batch_size, input, pi, v);
}

static void tensorrt_predict(NetworkWrapper* nw, int batch_size, float *input, float *pi, float *v) {
    if(batch_size != PREDICT_BUFFER_SIZE) {
        exit(-1); 
    }
    TensorRTComponent* trt = nw->tensorrt_component; 
    trt->predict(PREDICT_BUFFER_SIZE, input, pi, v);
}

int main() {
    int thread_num = 100;
    printf("start\n");
    fflush(stdout);
    float input_dataset[1000 * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float pi_dataset[1000 * BOARD_SIZE * BOARD_SIZE];
    float v_dataset[1000];
    float pi[1000 * BOARD_SIZE * BOARD_SIZE];
    float v[1000];
    NetworkWrapper nw = NetworkWrapper();
    nw.save_weights("/hy-tmp/mlzeros/mlzero/ckpt/weight_0.ckpt"); 
    nw.load_weight("/hy-tmp/mlzeros/mlzero/ckpt/weight_0.ckpt");
    
    float other_input[PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE];
    float other_pi[PREDICT_BUFFER_SIZE * BOARD_SIZE * BOARD_SIZE];
    float other_v[PREDICT_BUFFER_SIZE];
    float other_pi_trt[PREDICT_BUFFER_SIZE * BOARD_SIZE * BOARD_SIZE];
    float other_v_trt[PREDICT_BUFFER_SIZE];

    for(int i = 0; i < PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
        other_input[i] = rand() % 2;
    }

    nw.predict_single(1, other_input, other_pi, other_v);
    
    // print otherinput
    printf("other input:\n");
    for(int i = 0; i < PLANES_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            for(int k = 0; k < BOARD_SIZE; k++) {
                printf("%.5f ", other_input[i * BOARD_SIZE * BOARD_SIZE + j * BOARD_SIZE + k]);
            }
            printf("\n");
        }
    }

    nw.predict_only_buffer(1, other_input, other_pi_trt, other_v_trt);

    printf("other pi:\n"); 
    for(int i = 0; i < BOARD_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            printf("%.5f ", other_pi[i * BOARD_SIZE + j]);
        }
        printf("\n");
    }
    printf("other v: %.5f\n", other_v[0]);
    fflush(stdout);

    printf("other pi_trt:\n");
    for(int i = 0; i < BOARD_SIZE; i++) {
        for(int j = 0; j < BOARD_SIZE; j++) {
            printf("%.5f ", other_pi_trt[i * BOARD_SIZE + j]);
        }
        printf("\n");
    }
    printf("other v_trt: %.5f\n", other_v_trt[0]);


    // printf("generate input data\n");
    // fflush(stdout);
    // for(int i = 0; i < 1000*PLANES_SIZE * BOARD_SIZE * BOARD_SIZE; i++) {
    //     input_dataset[i] = rand()%2;
    // }

    // printf("start predict\n");
    // fflush(stdout);
    // printf("start predict_single\n");
    // fflush(stdout);
    // for(int i = 0; i < thread_num; i++) {
    //     nw.predict_single(1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi_dataset + i * BOARD_SIZE * BOARD_SIZE, v_dataset + i);
    // }

    // printf("start predict buffer\n");
    // fflush(stdout);
    // std::thread* threads[thread_num];
    // for(int i = 0; i < thread_num; i++) {
    //     // nw.predict_test(1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //     threads[i] = new std::thread(predict_operation_test, &nw, 1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //     // threads[i] = new std::thread(tensorrt_predict, &nw, PREDICT_BUFFER_SIZE, input_dataset + i * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * PREDICT_BUFFER_SIZE * BOARD_SIZE * BOARD_SIZE, v + i * PREDICT_BUFFER_SIZE);
    // }
    // for(int i = 0; i < thread_num; i++) {
    //     threads[i]->join();
    //     delete threads[i];
    // }

    // Timer timer = Timer();
    // timer.start();
    // for(int times = 0; times < 10; times++) {
    //     std::thread* threads[thread_num];
    //     for(int i = 0; i < thread_num; i++) {
    //         // nw.predict_test(1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //         threads[i] = new std::thread(predict_operation_trt, &nw, 1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //         // threads[i] = new std::thread(tensorrt_predict, &nw, PREDICT_BUFFER_SIZE, input_dataset + i * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * PREDICT_BUFFER_SIZE * BOARD_SIZE * BOARD_SIZE, v + i * PREDICT_BUFFER_SIZE);
    //     }
    //     for(int i = 0; i < thread_num; i++) {
    //         threads[i]->join();
    //         delete threads[i];
    //     }
    // }
    // timer.end();
    // printf("trt_time: %f\n", timer.get_duration_us()/1000.0);
    // fflush(stdout);

    // Timer timer_2 = Timer();
    // timer_2.start();
    // for(int times = 0; times < 10; times++) {
    //     std::thread* threads[thread_num];
    //     for(int i = 0; i < thread_num; i++) {
    //         // nw.predict_test(1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //         threads[i] = new std::thread(predict_operation_test, &nw, 1, input_dataset + i * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * BOARD_SIZE * BOARD_SIZE, v + i);
    //         // threads[i] = new std::thread(tensorrt_predict, &nw, PREDICT_BUFFER_SIZE, input_dataset + i * PREDICT_BUFFER_SIZE * PLANES_SIZE * BOARD_SIZE * BOARD_SIZE, pi + i * PREDICT_BUFFER_SIZE * BOARD_SIZE * BOARD_SIZE, v + i * PREDICT_BUFFER_SIZE);
    //     }
    //     for(int i = 0; i < thread_num; i++) {
    //         threads[i]->join();
    //         delete threads[i];
    //     }
    // }
    // timer_2.end();
    // printf("test_time: %f\n", timer_2.get_duration_us()/1000.0);
    // fflush(stdout);
    

    // //check
    // for(int i = 0; i < thread_num; i++) {
    //     for(int j = 0; j < BOARD_SIZE * BOARD_SIZE; j++) {
    //         if(fabs(pi_dataset[i * BOARD_SIZE * BOARD_SIZE + j] - pi[i * BOARD_SIZE * BOARD_SIZE + j]) > 0.05) {
    //             printf("error, i: %d, j: %d, pi_dataset: %f, pi: %f\n", i, j, pi_dataset[i * BOARD_SIZE * BOARD_SIZE + j], pi[i * BOARD_SIZE * BOARD_SIZE + j]);

    //             //print this pi
    //             printf("pi_dataset:\n");
    //             for(int k = 0; k < BOARD_SIZE; k++) {
    //                 for(int l = 0; l < BOARD_SIZE; l++) {
    //                     printf("%.2f ", pi_dataset[i * BOARD_SIZE * BOARD_SIZE + k * BOARD_SIZE + l]);
    //                 }
    //                 printf("\n");
    //             }
    //             printf("pi:\n");
    //             for(int k = 0; k < BOARD_SIZE; k++) {
    //                 for(int l = 0; l < BOARD_SIZE; l++) {
    //                     printf("%.2f ", pi[i * BOARD_SIZE * BOARD_SIZE + k * BOARD_SIZE + l]);
    //                 }
    //                 printf("\n");
    //             }
    //             return 0;
    //         }
    //     }
    //     if(fabs(v_dataset[i] - v[i]) > 0.05) {
    //         printf("error, i: %d, v_dataset: %f, v: %f\n", i, v_dataset[i], v[i]);
    //         //print this pi
    //         printf("pi_dataset:\n");
    //         for(int k = 0; k < BOARD_SIZE; k++) {
    //             for(int l = 0; l < BOARD_SIZE; l++) {
    //                 printf("%.2f ", pi_dataset[i * BOARD_SIZE * BOARD_SIZE + k * BOARD_SIZE + l]);
    //             }
    //             printf("\n");
    //         }
    //         printf("pi:\n");
    //         for(int k = 0; k < BOARD_SIZE; k++) {
    //             for(int l = 0; l < BOARD_SIZE; l++) {
    //                 printf("%.2f ", pi[i * BOARD_SIZE * BOARD_SIZE + k * BOARD_SIZE + l]);
    //             }
    //             printf("\n");
    //         }
    //         return 0;
    //     }
    // }

    printf("success\n");
    fflush(stdout);
}


// int main() {
//     printf("start\n");
//     Timer timer_cpu = Timer();
//     Timer timer_gpu = Timer();
//     NetworkWrapper nw = NetworkWrapper();
//     nw.load_weight("/home/work/file/mlzero/test/ckpt/latest.ckpt");
//     auto net = nw.get_net();
//     long long cpu_time[1024] = {0}; 
//     long long gpu_time[1024] = {0}; 

//     int steps = 100;
//     int n = steps;
//     while(n--) {
//         //cpu
//         net->to(torch::kCPU);
//         //warm up
//         for(int i = 1; i < 100; i++) {
//             torch::Tensor input_tensor = torch::rand({1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
//             net->forward(input_tensor);
//         }
//         for(int i = 1; i <= 512; i *= 2) {
//             printf("batchsize: %d\n", i);
//             torch::Tensor input_tensor = torch::rand({i, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
//             timer_cpu.start();
//             auto outputs = net->forward(input_tensor);
//             timer_cpu.end();
//             printf("batchsize: %d, cpu time: %f\n", i, timer_cpu.get_duration_us()/1000.0);
//             fflush(stdout);
//             cpu_time[i] += timer_cpu.get_duration_us()/1000.0;
//         }

//         //gpu
//         net->to(torch::kCUDA);
//         //warm up
//         for(int i = 1; i < 100; i++) {
//             torch::Tensor input_tensor = torch::rand({1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE}).to(torch::kCUDA);
//             net->forward(input_tensor);
//         }
//         for(int i = 1; i <= 512; i *= 2) {
//             printf("batchsize: %d\n", i);
//             torch::Tensor input_tensor = torch::rand({i, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE});
//             timer_gpu.start();
//             torch::Tensor input_tensor_gpu = input_tensor.to(torch::kCUDA);
//             auto outputs_gpu = net->forward(input_tensor_gpu);
//             torch::Tensor policy_tensor_gpu = std::get<0>(outputs_gpu).to(torch::kCPU);
//             torch::Tensor value_tensor_gpu = std::get<1>(outputs_gpu).to(torch::kCPU);
//             timer_gpu.end();
//             printf("batchsize: %d, gpu time: %f\n", i, timer_gpu.get_duration_us()/1000.0);
//             fflush(stdout);
//             gpu_time[i] += timer_gpu.get_duration_us()/1000.0;
//         }
//     }

//     for(int i = 1; i <= 512; i *= 2) {
//         printf("batchsize: %d, cpu time: %f, gpu time: %f\n", i, cpu_time[i]/100.0, gpu_time[i]/100.0);
//         fflush(stdout);
//     }

