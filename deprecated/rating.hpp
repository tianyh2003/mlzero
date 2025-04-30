#ifndef MLZERO_RATING_HPP
#define MLZERO_RATING_HPP

#include <cmath>
#include <stdexcept>

// 计算K因子的函数
int get_k_factor(const float* ratings, size_t size) {
    if (ratings == nullptr || size == 0) {
        throw std::invalid_argument("Invalid ratings array");
    }

    bool all_below_2100 = true;
    bool all_below_2400 = true;
    bool any_above_2100 = false;
    bool all_above_2400 = true;

    for (size_t i = 0; i < size; i++) {
        if (ratings[i] >= 2100) all_below_2100 = false;
        if (ratings[i] >= 2400) all_below_2400 = false;
        if (ratings[i] >= 2100) any_above_2100 = true;
        if (ratings[i] < 2400) all_above_2400 = false;
    }

    if (all_below_2100) return 32;
    if (all_below_2400 && any_above_2100) return 24;
    if (all_above_2400) return 16;
    return 32; // 默认值
}

class EloRating {
private:
    float rating;

public:
    // 构造函数，默认初始化为0分
    explicit EloRating(float initial_rating = 0.0f) : rating(initial_rating) {}

    // 获取当前评分
    float get_rating() const { return rating; }

    // 计算预期得分
    float expected_score(float opponent_rating) const {
        return 1.0f / (1.0f + std::pow(10.0f, (opponent_rating - rating) / 400.0f));
    }

    // 更新评分
    void update_rating(float opponent_rating, float actual_score) {
        const float ratings[] = {rating, opponent_rating};
        const int k = get_k_factor(ratings, 2);
        
        const float expected = expected_score(opponent_rating);
        rating += k * (actual_score - expected);
    }

    void set_rating(float _rating) {
        rating = _rating;
    }
};

#endif

// // 示例用法
// int main() {
//     try {
//         // 初始化两个玩家
//         EloRating player1(1500.0f);
//         EloRating player2(1500.0f);

//         // 进行10次模拟对战
//         for (int i = 0; i < 10; i++) {
//             const bool player1_wins = (i % 2 == 0);

//             EloRating& winner = player1_wins ? player1 : player2;
//             EloRating& loser = player1_wins ? player2 : player1;

//             // 更新评分
//             winner.update_rating(loser.get_rating(), 1.0f);
//             loser.update_rating(winner.get_rating(), 0.0f);

//             // 输出当前评分（示例中简化输出）
//             // printf("Player 1 rating: %.1f\n", player1.get_rating());
//             // printf("Player 2 rating: %.1f\n", player2.get_rating());
//         }
//     } catch (const std::exception& e) {
//         // 处理异常
//     }

//     return 0;
// }