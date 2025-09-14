#include "Max_clique.hpp"

MaxClique::MaxClique(int V, vector<vector<int>> g)
{
    this->g = g;
    n = V-1;
    cnt = vector<int>(V);
    vis = vector<int>(V);
    ans = 0;
}

vector<int> MaxClique::solve()
{
	for (int i = n; i >= 0; i--){
		vis[0] = i;
		dfs(i, 1);
		cnt[i] = ans;
    }
    return answer;
}

bool MaxClique::dfs(int cur, int num) {
	for (int i = cur + 1; i <= n; i++) {
		if (cnt[i] + num <= ans)
			return 0;
		if (g[cur][i])
		{
			int ok = 1;
			for (int j = 0; j < num; j++)
				if (!g[i][vis[j]])
				{
					ok = 0;
					break;
				}
			if (ok) {
				vis[num] = i;
				if (dfs(i, num + 1))
					return 1;
			}
		}
	}
	if(num >ans){
        ans = num;
        answer.clear();
        for(int i=0; i<ans; i++){
            answer.emplace_back(vis[i]);
        }
    }
	return (ans == max(num, ans) ? 0 : 1);
}