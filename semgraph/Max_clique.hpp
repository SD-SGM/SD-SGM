#pragma once
#include<bits/stdc++.h>
using namespace std;

class MaxClique{
    public:
    MaxClique(int V, vector<vector<int>> g);
    vector<int> solve();
    private:
    bool dfs(int cur, int num);
    vector<vector<int>> g;
    vector<int> vis;
    vector<int> cnt;
    int ans, n;
    vector<int> answer;
};