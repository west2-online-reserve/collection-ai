#include <iostream>
#include <iomanip>
#include <windows.h>
#include <algorithm>
using namespace std;

int n;
int map[30][30];

int tmp_map[30][30], last_map[30][30], flag[30][30][30], color_num[30], finded_color[30];
// 行为 1 ，列为 2
int row_and_col_flag[30][2][30], row_and_col_num[30][2];
int count_num = 1, dfs_num;

void cout_map() {
    for (int i=1;i <= n;i++){
        for (int j=1;j <= n;j++)
            cout << setw(3) << map[i][j];
        cout << endl;
    }
    return;
}


void count_color() {
    // 清空所有颜色的计数
    for (int i=1;i <= n;i++)
        color_num[i] = 0;
    // 遍历计数
    for (int i=1;i <= n;i++) {
        for (int j=1;j <= n;j++)
            if (map[i][j] != 0)
                color_num[map[i][j]]++;
    }
    return;
}

void draw_row_and_col() {
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++) {
            row_and_col_flag[map[i][j]][0][j] = 1;
            row_and_col_flag[map[i][j]][1][i] = 1;
        }
    return;
}

void count_row_and_col() {
    for (int k=1;k <= n;k++){
        row_and_col_num[k][0] = 0;
        row_and_col_num[k][1] = 0;
    }
    for (int k=1;k <= n;k++){
        for (int i=1;i <= n;i++){
            if (row_and_col_flag[k][0][i] == 1)
                row_and_col_num[k][0]++;
            if (row_and_col_flag[k][1][i] == 1)
                row_and_col_num[k][1]++;
        }
    }
    return;
}

void draw_map(int x, int y) {
    int num = map[x][y];
    for (int i=1;i <= n;i++) {
        map[x][i] = 0;
        map[i][y] = 0;
    }
    map[x-1][y-1] = 0;
    map[x-1][y+1] = 0;
    map[x+1][y-1] = 0;
    map[x+1][y+1] = 0;
    map[x][y] = num;
    finded_color[map[x][y]] = 1;
    return;
}

bool catch_one_color_one() {
    count_color();
    for (int k=1;k <= n;k++)
        if (color_num[k] == 1 && finded_color[k] == 0) {
            for (int i=1;i <= n;i++)
                for (int j=1;j <= n;j++)
                    if (map[i][j] == k){
                        draw_map(i, j);
                        return true;
                    }
        }
    
    return false;
}

bool catch_just_one_in_line() {
    bool find_flag = false;
    for (int k=1;k <= n;k++) {
        int row_num=0, col_num=0;
        // 检验第 k 行中是否有单独的块
        for (int j=1;j <= n;j++){
            if (map[k][j] != 0){
                if (col_num != 0){
                    col_num = 0;
                    break;
                }
                else {
                    col_num = j;
                }
            }
        }
        if (col_num != 0 && finded_color[map[k][col_num]] == 0) {
            draw_map(k, col_num);
            find_flag = true;
        }
        // 检验第 k 列中是否有单独的块
        for (int i=1;i <= n;i++){
            if (map[i][k] != 0){
                if (row_num != 0){
                    row_num = 0;
                    break;
                }
                else {
                    row_num = i;
                }
            }
        }
        if (row_num != 0 && finded_color[map[row_num][k]] == 0) {
            draw_map(row_num, k);
            find_flag = true;
        }
    }
    return find_flag;
}

bool exclude_one_line_color() {
    bool exclude_flag=false;
    for (int k=1;k <= n;k++){
        if (finded_color[k] == 1)
            break;
        if (row_and_col_num[k][0] == 1){
            for (int j=1;j <= n;j++)
                if (row_and_col_flag[k][0][j] == 1){
                    for (int i=1;i <= n;i++)
                        if (map[i][j] != k && map[i][j] != 0){
                            map[i][j] = 0;
                            exclude_flag = true;
                        }
                    break;
                }
        }
        if (row_and_col_num[k][1] == 1){
            for (int i=1;i <= n;i++)
                if (row_and_col_flag[k][1][i] == 1){
                    for (int j=1;j <= n;j++)
                        if (map[i][j] != k && map[i][j] != 0){
                            map[i][j] = 0;
                            exclude_flag = true;
                        }
                }
        }
    }
    return exclude_flag;
}

void exclude() {
    int num=0;
    while (num <= 10) {
        num++;
        if (catch_one_color_one())
            continue;
        if (catch_just_one_in_line())
            continue;
        draw_row_and_col();
        count_row_and_col();

        if (exclude_one_line_color())
            continue;
        return;
    }
    return;
}



void add_flag(int x, int y, int num) {
    for (int i=1;i <= n;i++) {
        flag[i][y][num] = 99;
        flag[x][i][num] = 99;
    }
    flag[x-1][y-1][num] = 99;
    flag[x-1][y+1][num] = 99;
    flag[x+1][y-1][num] = 99;
    flag[x+1][y+1][num] = 99;
    flag[x][y][num] = -num;
    return;
}

void clear_flag(int x, int y, int num) {
    for (int i=1;i <= n;i++) {
        flag[i][y][num] = 0;
        flag[x][i][num] = 0;
    }
    flag[x-1][y-1][num] = 0;
    flag[x-1][y+1][num] = 0;
    flag[x+1][y-1][num] = 0;
    flag[x+1][y+1][num] = 0;
    return;
}

void draw_tmp_map() {
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            tmp_map[i][j] = map[i][j];
    
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++)
            for (int k=1;k <= n;k++) {
                if (flag[i][j][k] != 0) {
                    tmp_map[i][j] = -1;
                    break;
                }
            }
    return;
}

void draw_last_map() {
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++) {
            for (int k=1;k <= n;k++)
                if (flag[i][j][k] < 0)
                    last_map[i][j] = 1;
            if (tmp_map[i][j] == n) {
                last_map[i][j] = 1;
            }

        }
    return;
}


void init() {
    cin >> n;
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++)
            cin >> map[i][j];
    count_color();
    int order_num[30], used_time[30]={0};
    for (int i=1;i <= n;i++)
        order_num[i] = color_num[i];
    sort(order_num, order_num + n);
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++)
            if(color_num[i] == order_num[j] && used_time[j] == 0){
                color_num[i] = j;
                used_time[j] = 1;
            }
    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++)
            map[i][j] = color_num[map[i][j]];
    return;
}

int dfs(int x, int y, int num) {
    if (num == n)
        return n;

    add_flag(x, y, num);
    draw_tmp_map();

    int goal = num + 1;
    bool find_goal = false;

    for (int i=1;i <= n;i++)
        for (int j=1;j <= n;j++) {
            draw_tmp_map();
            if (tmp_map[i][j] == goal)
                if (dfs (i, j, goal) == n)
                    return n;
        }

    clear_flag(x, y, num);

    return num;
}

int main() {
    init();
    SetConsoleOutputCP(CP_UTF8);
    exclude();
    bool exit_flag = false;

    // 暴力搜索出结果
    for (int i=1;i <= n;i++){
        for (int j=1;j <=n;j++)
            if(map[i][j] == 1){
                dfs_num = 0;
                count_num++;
                // 初始状态允许一只 初始 num=1
                if (dfs(i, j, 1) == n) {
                    draw_last_map();
                    exit_flag = true;
                    break;
                }
                else {
                    map[i][j] = 0;
                    exclude();
                }
            }
        if(exit_flag) break;
    }

    // 输出
    for (int i=1;i <=n ;i++) {
        for (int j=1;j <= n;j++)
            cout << last_map[i][j] << ' ';
        cout << endl;
    }
    return 0;
}