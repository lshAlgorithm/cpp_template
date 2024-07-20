// Vittual tree solved for "P_2495_SDOI_2011_消耗战" on luogu

#include<bits/stdc++.h>
using namespace std;

// #define _DEBUG
#define fi first
#define se second
#define all(a) a.begin(),a.end()
#ifdef _DEBUG
#define debug(a) cout << #a << '=' << a << endl;
#else
#define debug(a) ;
#endif
#define eb emplace_back
#define pb push_back
#define int long long

typedef long long LL;
typedef pair<int, int> pii;

const LL inf = 0x3f3f3f3f3f3f3f3f;

bool Mbe;

const int N = 2.5e5 + 10, M = 5e5 + 10;
int h[N], e[N << 1], ne[N << 1], val[N << 1], pre[N][25], dep[N], dfn[N], cur, cnt;
int h2[N], e2[N << 1], ne2[N << 1], cur2;
bitset<M> need;
int minv[N];

void add(int a, int b, int w) {
    e[cur] = b,  ne[cur] = h[a], h[a] = cur, val[cur++] = w;
}

void add2(int a, int b) {
    e2[cur2] = b, ne2[cur2] = h2[a], h2[a] = cur2++;
}

void dfs(int u, int f) {
    if(u == 1) dep[1] = 1, minv[1] = inf;
    dfn[u] = ++cnt;
    for(int i = h[u]; ~i; i = ne[i]) {
        int node = e[i];
        // debug(node)
        if(node == f) continue;
            dep[node] = dep[u] + 1;
            pre[node][0] = u;
            // what we need!
            minv[node] = min(minv[u], val[i]);
            dfs(node, u);
    }
}

int lca(int a, int b) {
    if(dep[a] < dep[b]) swap(a, b);
    // debug(dep[a]) debug(dep[b])
    for(int i = 22; ~i; i--) {
        // debug(dep[pre[a][i]])
        if(dep[pre[a][i]] >= dep[b]) {
            a = pre[a][i];
        }
    }
        
    if(a == b) {
        return a;
    }

    for(int i = 22; ~i; i--) {
        if(pre[a][i] != pre[b][i]) {
            a = pre[a][i];
            b = pre[b][i];
        }
    }
    return pre[a][0];
}

bool cmp(int a, int b) {
    return dfn[a] < dfn[b];
}

LL dfs2(int u) {
    LL res = 0;
    for(int i = h2[u]; ~i; i = ne2[i]) {
        int node = e2[i];
        // if(node == f) continue;
        res += dfs2(node);
    //     debug(node)
    // //    debug(val2[i])
    //     debug(val2[i])
    //    dp[u] += (st.count(node) ? val2[i] : min(val2[i], dp[node]));
    }
    if(need[u]) {
        res = minv[u];
    }
    else {
        res = min(res, 1LL * minv[u]);
    }
    // ayeeeeeeeeeeeeeeeeeeeeee
    need[u] = 0;
    h2[u] = -1;
    debug(u)
    return res;
}
void solve() {
    #define only_one
    int n; cin >> n;
    memset(h, -1, sizeof h);
    
    for (int i = 0; i < n - 1; i++) {
        int u, v, w; cin >> u >> v >> w;
        add(u, v, w);
        add(v, u, w);
    }

    dfs(1, -1);
    
    //debug("okdfs")
    for(int i = 1; i <= 22; i++) {
        for(int j = 1; j <= n; j++) {
    //        debug(i) debug(j)
            pre[j][i] = pre[pre[j][i - 1]][i - 1];
        }
    }
    //debug("ok")
    int m; cin >> m;
        memset(h2, -1, sizeof(int) * (n + 5));//the number of nodes must be within boundary
    while(m--) {
        // why is it this place???
        int num; cin >> num;
        cur2 = 0;
        //debug(num)
        vector<int> q(num);
        for(auto &c: q) cin >> c, need[c] = 1;
        q.push_back(1);

        sort(q.begin(), q.end(), cmp);
        vector<int> tmp;
        for(int i = 0; i < q.size() - 1; i++) {
            tmp.eb(q[i]);
            int l = lca(q[i], q[i + 1]);
            tmp.eb(l);
        }
        tmp.eb(q.back());
        sort(tmp.begin(), tmp.end(), cmp);
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
        for(auto c: tmp) debug(c)
        // for(auto c: q) debug(c)
        for(int i = 0; i < tmp.size() - 1; i++) {
            int l = lca(tmp[i], tmp[i + 1]);
            //debug(l) debug(q[i + 1])
            add2(l, tmp[i + 1]);//if you add the weight, you are completely ignorant of trDP
        }
        cout << dfs2(1) << '\n';
    }
}

// bool Med;
signed main() {
    std::ios::sync_with_stdio(false);
    //fprintf(stderr, "%.3lf MB\n", (&Med - &Mbe) / 1048576.0);
    cin.tie(0);
    int T = 1; 
#ifndef only_one
    cin >> T;
#endif
    while(T--) solve();
    return 0;
}
