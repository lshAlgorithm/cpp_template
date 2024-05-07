## 倍增思想实现`LCA()`最近公共祖先 [洛谷P3379](https://www.luogu.com.cn/problem/P3379)
```cpp
#include<bits/stdc++.h>
using namespace std;

//#define _DEBUG
#define For(i, begin, end) for (__typeof(begin) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); (begin) > (end) ? i-- : i++)
typedef long long LL;

const int N = 5e5 + 10, M = N << 1;
int h[N], e[M], ne[M], cur;
int dep[N], pre[N][25];//pre存储节点编号

void add(int a, int b) {
    e[cur] = b; ne[cur] = h[a]; h[a] = cur++;
    //printf("build tree%d\t", cur);
}

//dfs
//典型前序遍历
void get_depth(int r) {
    for(int i = h[r]; i != -1; i = ne[i]) {
        int node = e[i];
        if(dep[node] == 0){//!
            dep[node] = dep[r] + 1;
            pre[node][0] = r;
            get_depth(node);
        }
    }
}

int Lca(int a, int b) {
    if(dep[a] < dep[b]) swap(a, b);
    For(i, 21, 0) {
        if(dep[pre[a][i]] >= dep[b]) {
            a = pre[a][i];
        }
    }
    if(a == b) return a;
    For(i, 21, 0) {
        if(pre[a][i] != pre[b][i]) {//判断父亲！
            a = pre[a][i];
            b = pre[b][i];
        }
    }
    return pre[a][0];//返回父亲！
}
int main() {
    // freopen("C:\\data\\inluogu.txt","r", stdin);
    // freopen("C:\\data\\outluogu.txt","w", stdout);//覆盖
    memset(h, -1, sizeof h);
    int n, m, s;
    scanf("%d%d%d", &n, &m, &s);
    For(i, 0, n - 1) {
        int a, b; scanf("%d%d", &a, &b);
        add(a, b); add(b, a);
    }

    dep[s] = 1;
    //pre[s][0] = 0;
    get_depth(s);

    //core code!
    For(i, 1, 21) {//起点终点！pre[x][0]都已赋值了，不用再算了
        For(j, 1, n + 1) {
            pre[j][i] = pre[pre[j][i - 1]][i - 1];
        }
    }
    while(m--) {
        int a, b; scanf("%d%d", &a, &b);
        printf("%d\n", Lca(a, b));
    }
    // fclose(stdin);
    // fclose(stdout);
    return 0;
}
```


## ST表---查询静态区间的最值 [洛谷P3865](https://www.luogu.com.cn/problem/P3865)
```cpp
#include<bits/stdc++.h>
using namespace std;

//#define _DEBUG
#define For(i, begin, end) for (__typeof(begin) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); (begin) > (end) ? i-- : i++)
typedef long long LL;

const int N = 1e6 + 10, M = 25;
int n;
int maxx[N][M], a[N];//[N][M]

inline void init() {
    For(i, 0, n) {
        maxx[i][0] = a[i];
    }
    for(int j = 1; (1 << j) <= n; j++) {//of size 2 ^ j ps.consistence matters
        for(int i = 0; (i + (1 << j) - 1) < n; i++) {
            maxx[i][j] = max(maxx[i][j - 1], maxx[i + (1 << (j - 1))][j - 1]);
        }
    }
}

inline int query(int l, int r) {
    int j = (int) log2(r - l + 1);
    return max(maxx[l][j], maxx[r - ((1 << j) - 1)][j]);//+1？？
}

int main() {
    // freopen("C:\\data\\inluogu.txt","r", stdin);
    // freopen("C:\\data\\outluogu.txt","w", stdout);//覆盖
    int m;
    scanf("%d%d", &n, &m);
    For(i, 0, n) {
        scanf("%d", &a[i]);
    }
    init();
    while(m--) {
        int l, r; scanf("%d%d", &l, &r);
        l--; r--;
        printf("%d\n", query(l, r));
    }
    // fclose(stdin);
    // fclose(stdout);
    return 0;
}
```

## 线段树（无懒标记）[洛谷P4145](https://www.luogu.com.cn/problem/P4145)
### 思考：
1. 本质上还是 __树__ ，
```cpp
#include<bits/stdc++.h>
#define tra(i, x, y) for (int i = x; i <= y; i++)
using namespace std;

//l, r 对应的是原数组的下标，u则是线段树的

// #define ls k >> 1
// #define rs k >> 1 | 1
const int N = 1e5 + 10;
long long q[N];
struct node{
    int l, r;
    long long v, sum;
}tr[N << 2];

void up (node &u, node &ls, node &rs) {
    u.sum = ls.sum + rs.sum;
    u.v = max(ls.v, rs.v);
}

void up (int u) {
    up(tr[u], tr[u << 1], tr[u << 1 | 1]);
}

void build(int u, int l, int r) {
    tr[u] = {l, r};
    if(l == r) {
        tr[u].sum = tr[u].v = q[l];
        return  ;
    }
    int mid = l + r >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    up(u);
}

//l, r绝对不能改变
void change(int u, int ll, int rr, int l, int r) {
    if (ll == rr) {//ll >= l && rr =< r?
        tr[u].sum = tr[u].v = sqrt(tr[u].v);
        return ;
    }
    int mid = ll + rr >> 1;

    //剪枝很重要捏
    if(mid >= l && tr[u << 1].v > 1) change(u << 1, ll, mid, l ,r);
    if(mid < r && tr[u << 1 | 1].v > 1) change(u << 1 | 1, mid + 1, rr, l, r);
    up(u);
}

long long query(int u, int ll, int rr, int l, int r) {
    if(ll >= l && rr <= r) return tr[u].sum;
    int mid = ll + rr >> 1;
    long long res = 0;
    if(l <= mid) res += query(u << 1, ll, mid, l ,r);
    if(r > mid) res += query(u << 1 | 1, mid + 1, rr, l, r);
    //cout << res << endl;
    return res;
    
}
int main() {
    //struct be initialized can be omitted
    //memset(tr, 0, sizeof tr);
    //printf("%lld", 1000000000 + 99900000000);
    int n; scanf("%d", &n);
    tra(i, 1, n) scanf("%lld", &q[i]);
    build(1, 1, n);
    int m, k, l, r;
    scanf("%d", &m);
    while(m--) {
        scanf("%d%d%d", &k, &l, &r);
        if(l > r) swap(l, r);
        if(!k) {
            change(1, 1, n, l, r);
            //cout << tr[1].sum << " " << tr[1].v << endl << endl;
        }
        else {printf("%lld\n", query(1, 1, n, l, r));}
    }
    return 0;
}
```

## DFS经典例题——火车进站[ACWING129](https://www.acwing.com/problem/content/description/131/)
### [参考题解](https://www.acwing.com/solution/content/27189/)
```cpp
#include <bits/stdc++.h>
using namespace std;
#define LL long long
#define For(i, x, y) for(__typeof(x) i = (x) - ((x) > (y)); i != (y) - ((x) > (y)); (x) > (y) ? i-- : i++)
#define  pb push_back

int n;
int remain, stk[25], tt;
vector<int> path;

inline void dfs(int u) {
    if(u == n + 1) {
        if(++remain > 20) exit(0);
        for(auto c : path) printf("%d", c);
        For(i, tt, 0) {printf("%d", stk[i]);}//这里不能出栈，一定要把栈保持着！
        puts("");
        return ;
    }
    if(tt) {//我愿称之为——先吐再进
        path.pb(stk[--tt]);
        dfs(u);
        stk[tt++] = path.back();
        path.pop_back();
    }
    stk[tt++] = u;
    dfs(u + 1);
    tt--;
}
int main(){
    scanf("%d", &n);
    dfs(1);
    return 0;
}
```

## 后缀数组(进阶指南P78)[ACWING140](https://www.acwing.com/problem/content/description/142/)
```cpp
const int P=131,QWQ=1919810;
unsigned long long m[QWQ],p[QWQ],s[QWQ];
char str[QWQ];
int len;
int get(int l,int r)
{
    return m[r]-m[l-1]*p[r-l+1];
}
int gmcp(int a,int b)
{
    int l=0,r=len-max(a,b)+1,mid;
    while(l<r)
    {
        mid=(l+r+1)>>1;
        if(get(a,a+mid-1)==get(b,b+mid-1))  l=mid;
        else    r=mid-1;
    }
    return l;
}
bool cmp(int a,int b)
{
    int l=gmcp(a,b),av=a+l>len?INT_MIN:str[a+l],bv=b+l>len?INT_MIN:str[b+l];//越界只可能是a + l == len
    return av<bv;
}
int main()
{
    scanf("%s",str+1);
    len=strlen(str+1);
    p[0]=1;
    for(int i=1;i<=len;i++)
    {
        m[i]=m[i-1]*P+str[i];
        p[i]=p[i-1]*P;
        s[i]=i;
    }
    sort(s+1,s+len+1,cmp);
    for(int i=1;i<=len;i++) cout<<s[i]-1<<' ';
    cout<<"\n0";
    for(int i=2;i<=len;i++) cout<<' '<<gmcp(s[i],s[i-1]);
    return 0;
}
```

## 基数排序（radix sort）[详见](https://zhuanlan.zhihu.com/p/126116878?utm_id=0)
* 在$O(n)$的复杂度内完成的稳定排序，核心思想与桶排序相同
    > 虽然但是，我好想把他放到模板里，它太优雅了
```cpp
#include <bits/stdc++.h>
using namespace std;
#define LL long long

int maxbit(int data[], int n) //辅助函数，求数据的最大位数
{
    int maxData = data[0];      ///< 最大数
    /// 先求出最大数，再求其位数，这样有原先依次每个数判断其位数，稍微优化点。
    for (int i = 1; i < n; ++i)
    {
        if (maxData < data[i])
            maxData = data[i];
    }
    int d = 1;
    int p = 10;
    while (maxData >= p)
    {
        //p *= 10; // Maybe overflow
        maxData /= 10;
        ++d;
    }
    return d;
}
void radixsort(int data[], int n) //基数排序
{
    int d = maxbit(data, n);
    int *tmp = new int[n];
    int *count = new int[10]; //计数器
    int i, j, k;
    int radix = 1;
    for(i = 1; i <= d; i++) //进行d次排序
    {
        for(j = 0; j < 10; j++)
            count[j] = 0; //每次分配前清空计数器
        for(j = 0; j < n; j++)
        {
            k = (data[j] / radix) % 10; //统计每个桶中的记录数
            count[k]++;
        }
        for(j = 1; j < 10; j++)
            count[j] = count[j - 1] + count[j]; //将tmp中的位置依次分配给每个桶
        for(j = n - 1; j >= 0; j--) //将所有桶中记录依次收集到tmp中
        {
            k = (data[j] / radix) % 10;
            tmp[count[k] - 1] = data[j];
            count[k]--;
        }
        for(j = 0; j < n; j++) //将临时数组的内容复制到data中
            data[j] = tmp[j];
        radix = radix * 10;
    }
    delete []tmp;
    delete []count;
}

int main(){
    int n; scanf("%d", &n);
    int* q; q = new int[n];
    for(int i = 0; i < n; i++) scanf("%d", &q[i]);
    radixsort(q, n);
    for(int i = 0; i < n; i++) printf("%d ", q[i]);
    puts("");
    return 0;
}
```

## 后缀自动机（解决字符串匹配问题的神器）[hdu4622](https://acm.hdu.edu.cn/showproblem.php?pid=4622)
```cpp
const int inf = 0x3f3f3f3f, mod = 1e9 + 7;
const int N = 2e3 + 10;
char s[N];

int sz, last;
struct node{
    int son[26];//图上的单向实心方向，son[letter] = address
    int father;//后缀链, = address
    int len;//最长长度
}t[N << 1];//空间复杂度O(2n)

void newNode(int slen) {
    t[++sz].len = slen;
    t[sz].father = -1;//父类不确定，或者在init()中就是-1
    memset(t[sz].son, 0, sizeof(t[sz].son));
}

void init() {
    sz = -1, last = 0;
    newNode(0);
}

void insert(int c) {
    newNode(t[last].len + 1);//maxlen一定递增，比前一个最长的一定长1
    int p = last, cur = sz;//sz = last + 1
    while(p != -1 && !t[p].son[c]) t[p].son[c] = cur/*存下标*/, p = t[p].father;//~= i = ne[i]
    debug(p);
    if(p == -1) t[cur].father = 0;
    else {
        int q = t[p].son[c];//多跑一个，直到他的儿子没有son[c]，之后再去它的儿子，才是我们要修改的点q
        debug(q);
        if(t[q].len == t[p].len + 1) t[cur].father = q;
        else {
            newNode(t[p].len + 1);//有丝分裂，开始
            int nq = sz;//newq是新节点
            memcpy(t[nq].son, t[q].son, sizeof(t[q].son));//儿子都一样
            t[nq].father = t[q].father;//父亲继承，t[q].father更新为nq
            t[q].father = nq;
            t[cur].father = nq;//需要insert的那个结点的后缀链
            while(p >= 0 && t[p].son[c] == q){//向上更新,让nq替代q
                t[p].son[c] = nq, p = t[p].father;
            }
        }
    }
    last = cur;//更新last，后面调用的时候也是用last
    #ifdef _DEBUG
    for(int i = 0; i <= sz; i++) {
        for(int j = 0; j < 26; j++) {
            if(t[i].son[j]) {
                int st = i, ed = t[i].son[j];
                printf("%d // (%c) // %d", st, j + 'a', ed);
                printf(" father == %d; len == %d\n", t[ed].father, t[ed].len);
            }
        }
    }
    #endif
}
//vector<vector<int>> dp(len + 5, vector<int> (len + 5, 0));
int dp[N][N];//不用memset,每次都是从0开始用SAM算
inline void solve() {
    scanf("%s", s);
    int slen = strlen(s);
    for(int i = 0; i < slen; i++) {
        init();//不同的起点要构建不同的SAM
        debug(i);
        for(int j = i; j < slen; j++) {
            debug(j);
            insert(s[j] - 'a');
            //利用性质：每个endpos相同的等价类中存储的后缀串都是递增的，前面已经有过的自然不能多算，
            //只需要计算当前新增多少个，即减去后缀链的len
            dp[i][j] = dp[i][j - 1] + t[last].len - t[t[last].father].len;
            debug(t[last].len); debug(t[t[last].father].len);
        }
    }
    int Q, l, r; 
    scanf("%d", &Q);
    while(Q--) {
        scanf("%d%d", &l, &r);
        printf("%d\n", dp[--l][--r]);
    }
    #endif
}

int main() {
    int T = 1; 
    cin >> T;
    while(T--) solve();
    return 0;
}
```

## n个点之间距离最近的两个点 [divide and conquer luogu1257](https://www.luogu.com.cn/problem/P1257)
```cpp
#include <bits/stdc++.h>
using namespace std;
#define LL long long
#define pii pair<int, int>
#define fi first
#define se second

const double eps = 1e-8, INF = 1e20;
const int N = 1e4 + 10;
struct Point{double x, y;};//;
Point p[N], tmp_p[N];
int sgn(int x) {
    if(fabs(x) < eps) return 0;
    else return x < 0 ? -1 : 1;
}
bool cmpxy(Point A, Point B) {
    return sgn(A.x - B.x) < 0 || (sgn(A.x - B.x) == 0 && sgn(A.y - B.y) < 0);
}
bool cmpy(Point A, Point B) {return sgn(A.y - B.y) < 0;}
double distance(Point A, Point B) {
    return hypot(A.x - B.x, A.y - B.y);
}
double closePair(int left, int right) {
    double dis = INF;
    if(left == right) return dis;
    if(left + 1 == right) return distance(p[left], p[right]);
    int mid = left + right >> 1;
    double d1 = closePair(left, mid);
    double d2 = closePair(mid + 1, right);
    dis = min(d1, d2);
    int k = 0, cnt = 0;
    for(int i = left; i <= right; i++) {
        if(fabs(p[mid].x - p[i].x) <= dis) tmp_p[cnt++] = p[i];
    }
    sort(tmp_p, tmp_p + k, cmpy);
    for(int i = 0; i < cnt; i++) {
        for(int j = i + 1; j < cnt; j++) {
            if(tmp_p[j].y - tmp_p[i].y >= dis) break;
            dis = min(dis, distance(tmp_p[i], tmp_p[j]));
        }
    }
    return dis;
}
int main(){
    int n; scanf("%d", &n);
    for(int i = 0; i < n; i++) scanf("%lf%lf", &p[i].x, &p[i].y);
    sort(p, p + n, cmpxy);
    printf("%.4f\n", closePair(0, n - 1));
    return 0;
}
```


## 杜教筛
```cpp
#include<bits/stdc++.h>
using namespace std;

typedef long long LL;
const int N = 5e6 + 7;

// 这三个都是筛素数也要用到的
int cnt;
int prime[N];
bool vis[N];

int mu[N];
LL phi[N];

unordered_map<int, int> summu;
unordered_map<int, LL> sumphi;

void init() {
    //int cnt = 0;
    vis[0] = vis[1] = 1;
    mu[1] = phi[1] = 1;
    for(int i = 2; i < N; i++) {
        if(!vis[i]) {
            prime[cnt++] = i;
            phi[i] = i - 1;
            mu[i] = -1;
        }
        //prime会代劳！它是最小质因数
        for(int j = 0; j < cnt && i * prime[j] <N; j++) {
            vis[i * prime[j]] = 1;
            if(i % prime[j]) {
                mu[i * prime[j]] = -mu[i];
                phi[i * prime[j]] = phi[prime[j]] * phi[i];//积性函数
            }
            else {
                mu[i * prime[j]] = 0;
                phi[i * prime[j]] = phi[i] * prime[j];//可证，比较显然
                break;
            }
        }

    }   
    for(int i = 1; i < N; i++) {
        mu[i] += mu[i - 1];
        phi[i] += phi[i - 1];
    }
}

int gsum(int x) {return x;}

LL getsmu(int x) {
    if(x < N) return mu[x];
    if(summu[x]) return summu[x];//记忆化
    LL ans = 1;
    for(LL l = 2, r; l <= x; l = r + 1) {
        r = x / (x / l);
        ans -= (gsum(r) - gsum(l - 1)) * getsmu(x / l);//整块运算
    }
    return summu[x] = ans / gsum(1);
}

LL getsphi(int x) {
    if(x < N) return phi[x];
    if(sumphi[x]) return sumphi[x];
    LL ans = x * (1LL * x + 1) / 2;//纯纯杜教筛公式
    for(LL l = 2, r; l <= x; l = r + 1) {
        r = x / (x / l);
        ans -= (gsum(r) - gsum(l - 1)) * getsphi(x / l);
    }
    return sumphi[x] = ans / gsum(1);
}

int main() {
    init();
    int t; scanf("%d", &t);
    while(t--) {
        int n; scanf("%d", &n);
        printf("%lld %lld\n", getsphi(n), getsmu(n));
    }
    return 0;
}
```


## 拓扑排序按字典序输出
```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
using namespace std;

//#define _DEBUG
#define fi first
#define se second
#define all(a) a.begin(),a.end()
#ifdef _DEBUG
#define debug(a) cout << #a << '-' << a << endl;
#else
#define debug(a) ;
#endif
#define eb empalce_back

typedef long long LL;
typedef pair<int, int> pii;

const int inf = 0x3f3f3f3f, mod = 1e9 + 7;
int n, a[25], h[30], e[30], ne[30], cur, in[25], top[25];
bool vis[25];
void add(int a, int b) {
    e[cur] = b, ne[cur] = h[a], h[a] = cur++;
}
void dfs(int z, int cnt) {
    //puts("\n");
    top[cnt] = z;
    
    if(cnt == n - 1) {
        for(int i = 0; i <= cnt; i++) {
            printf("%c", top[i] + 'a');
        }
        puts("");
        return;
    }
    vis[z] = 1;
    for(int i = h[z]; ~i; i = ne[i]) { 
        int node = e[i];
        debug(z);
        debug(vis[24])
        debug(node)
        if(!vis[node]) {
            in[node] --;
        }
    }

    //brilliant!一定要遍历0~n
    for(int j = 0; j < n; j++) {
        int i = a[j];
        debug(in[24]);
        if(in[i] == 0 && !vis[i]) {
            // debug(i);
            dfs(i, cnt + 1); 
            
        }
    }

    for(int i = h[z]; ~i; i = ne[i]) { 
        int node = e[i];
        if(!vis[node]) {
            //puts("++");
            debug(node)
            in[node]++;
        }
    }
    vis[z] = 0;
    // puts("");
}
void solve() {
    char s[110];
    while(gets(s)) {
        n = 0;
        memset(h, -1, sizeof h);
        memset(in, 0, sizeof in);
        // memset(e, 0, sizeof e);
        // memset(ne, 0, sizeof ne);
        cur = 0;
        int len = strlen(s);
        for(int i = 0; i < len; i++) {
            if(s[i] >= 'a' && s[i] <= 'z') {
                a[n++] = s[i] - 'a';
            }
        }
        sort(a, a + n); //按字典序输出的重要语句！
        gets(s);
        len = strlen(s);
        bool flag = 1;
        for(int i = 0; i < len; i++) {
            int st, ed;
            if(flag && s[i] >= 'a' && s[i] <= 'z') {
                st = s[i] - 'a';
                flag = 0;
                continue;
            }
            if(!flag && s[i] >= 'a' && s[i] <= 'z') {
                ed = s[i] - 'a';
                flag = 1;
                add(st, ed);
                in[ed] ++;
                debug(st) debug(ed)
                continue;
            }
        }
        debug(in[21])
        // puts("map of z");
        // for(int i = h[25]; ~i; i = ne[i])
        //     debug(e[i]);
        // puts("end of map");
        for(int i = h[21]; ~i; i = ne[i]) debug(e[i]);
        for(int i = 0; i < n; i++) {
            if(!in[a[i]]) dfs(a[i], 0);
        }
        puts("");
    }
}

int main() {
    int T = 1; 
    // cin >> T;
    while(T--) solve();
    return 0;
}
```

## 欧拉回路 + 建模
```cpp
const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 1e5;
int num[N];
int st_edge[10 * N];
char st_ans[10 * N];
int top_s, top_a, m;

void no_dfs(int i) {
    int edge;
    while(num[i] < 10) {
        edge = 10 * i + num[i];
        num[i]++;
        st_edge[top_s++] = edge;
        // printf(" %02d -> ", i);
        i = edge % m; //continue to dfs
        // printf(" %02d: edge = %03d\n", i, edge);
    }
}
int main(){
    int n;
    while(scanf("%d", &n) && n) {
        m = 1;
        for(int i = 0; i < n - 1; i++) m *= 10;//number of edges
        for(int i = 0; i < m; i++) num[i] = 0;//statrt from 0
        no_dfs(0);
        int edge;
        while(top_s) {
            edge = st_edge[--top_s];
            st_ans[top_a++] = edge % 10 + '0';
            no_dfs(edge / 10); //backtracking!
        }
        for(int i = 1; i < n; i++) printf("0");
        while(top_a)
            printf("%c", st_ans[--top_a]);
        puts("");
    }
    return 0;
}
```

## tarjen求割点 poj1144
```cpp
const int MAXN=105;
vector<int> mp[MAXN];
int dfn[MAXN],low[MAXN],time;
bool crit[MAXN];
int root;
void dfs(int u,int fa)
{
    dfn[u]=low[u]=++time;
    int son=0;
    for(int i=0;i<mp[u].size();i++)
    {
        int v=mp[u][i];
        if(!dfn[v])
        {
            dfs(v,u);
            son++;
            low[u]=min(low[u],low[v]);
            if((root==u&&son>1)||(u!=root&&dfn[u]<=low[v]))
            {
                crit[u]=true;
            }
        }
        else if(v!=fa)    low[u]=min(low[u],dfn[v]);
    }
}

int n;
int main()
{
    while(scanf("%d",&n)!=EOF&&n!=0)
    {
        for(int i=1;i<=n;i++)
            mp[i].clear();
        memset(dfn,0,sizeof(dfn));
        memset(low,0,sizeof(low));
        time=0;
        memset(crit,false,sizeof(crit));
        int u;
        while(scanf("%d",&u)!=EOF&&u!=0)
        {
            int v;
            while(getchar()!='\n')
            {
                scanf("%d",&v);
                mp[u].push_back(v);
                mp[v].push_back(u);
            }
        }
        for(int i=1;i<=n;i++)
            if(!dfn[i])
            {
                root=i;
                dfs(i,-1);
            }
        int cnt=0;
        for(int i=1;i<=n;i++)
            if(crit[i])
                cnt++;
        printf("%d\n",cnt);    
        
    }
    return 0;
}
```

## tarjan判是否连通(hdu1269)
```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 1e4 + 5;
int cnt, low[N], num[N], dfn;
int sccno[N], stk[N], top;
int h[N], e[N << 1], ne[N << 1], cur;
// vector<int> G[N];

void add(int a, int b) {
    e[cur] = b, ne[cur] = h[a], h[a] = cur++;
}

void dfs(int u) {
    stk[top++] = u;
    low[u] = num[u] = ++dfn;
    for(int i = h[u]; ~i; i = ne[i]) {
    // for(int i = 0; i < G[u].size(); i++){
        int v = e[i];
        // int v = G[u][i];
        if(!num[v]) {
            dfs(v);
            low[u] = min(low[u], low[v]);
        }
        else if(!sccno[v]) {
            low[u] = min(low[u], num[v]);
        }
    }
    if(low[u] == num[u]) {
        cnt++;
        while(1) {
            int v = stk[--top];
            sccno[v] = cnt;
            if(u == v) break;
        }
    }
}
void tarjan(int n) {//SCC合并点
    cnt = top = dfn = 0;
    memset(sccno, 0, sizeof sccno);
    memset(num, 0, sizeof num);
    memset(low, 0, sizeof low);
    for(int i = 1; i <= n; i++) {
        if(!num[i])
            dfs(i);
    }
}
int main() {
    int n, m, u, v;
    while(~scanf("%d%d", &n, &m) && (n || m)) {
        // for(int i = 1; i <= n; i++) {G[i].clear();}
        memset(h, -1, 4 * (n + 1));
        cur = 0;
        for(int i = 0; i < m; i++) {
            scanf("%d%d", &u, &v);
            add(u, v);
            // G[u].push_back(v);
        }
        tarjan(n);
        printf("%s\n", cnt == 1 ? "Yes" : "No");
    }
    return 0;
}
```


## 基环树+树上dp[骑士](https://www.luogu.com.cn/problem/P2607)
```cpp
#include<bits/stdc++.h>
using namespace std;

#define _DEBUG
#define fi first
#define se second
#define all(a) a.begin(),a.end()
#ifdef _DEBUG
#define debug(a) cout << #a << '-' << a << endl;
#else
#define debug(a) 
#endif
#define eb emplace_back

typedef long long LL;
typedef pair<int, int> pii;

const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 1e6 + 100;
int value[N], father[N], mark;
LL dp[N][2];
bool vis[N];
vector<int> g[N];
void dfs(int u) {
    dp[u][0] = 0;
    dp[u][1] = value[u];
    vis[u] = 1;
    for(int v : g[u]) {
        //能确保递归会返回？
        if(v == mark) continue;//删去一条边
        dfs(v);
        dp[u][1] += dp[v][0];
        dp[u][0] += max(dp[v][0], dp[v][1]);
    }
}

int check(int u) {
    vis[u] = true;
    int f = father[u];
    if(vis[f]) return f;
    else return check(f);//往上回溯 一定能找到
}

LL solve(int u) {
    LL res = 0;
    mark = check(u);
    dfs(mark);
    res = max(dp[mark][0], res);
    mark = father[mark];
    dfs(mark);//没有上司的舞会  树上dp
    res = max(dp[mark][0], res);//总而言之 取max就完事了
    return res;
}

int main() {
    int n; scanf("%d", &n);
    for(int i = 1; i <= n; i++) {
        int d;
        scanf("%d%d", &value[i], &d);
        g[d].eb(i);
        father[i] = d;//只可能讨厌一个人，他的父亲
    }
    LL ans = 0;
    for(int i = 1; i <= n; i++) {
        if(!vis[i]) ans += solve(i);
    }
    printf("%lld\n", ans);
    return 0;
}
```

## 持久化Trie树[acwing256](https://www.acwing.com/problem/content/258/)
    1. 求l~r中一个数t，使得s[t] ^ s[t + 1] ^ ... ^ s[n] ^ x最大
    2. 前缀异或和+可持久化Trie树
```cpp
const int N = 2 * 3e5 + 10, M = 25 * N;//3e5个数，3e5次操作，最多N个root结点
int tr[M][2], max_id[M];//一共M个结点，N个root，深度最大为25(1e7)
int root[N], cur;
int s[N];
int n, m;

void insert(int i, int k, int p, int q) {
    if(k < 0) {
        max_id[q] = i;
        return;
    }
    int v = s[i] >> k & 1;
    if(p) tr[q][!v] = tr[p][!v];
    tr[q][v] = ++ cur;
    insert(i, k - 1, tr[p][v], tr[q][v]);//往深处递归
    max_id[q] = max(max_id[tr[q][v]], max_id[tr[q][!v]]);
}

int query(int now, int val, int limit) {
    int p = now;
    for(int i = 23; ~i; i--) {
        int v = val >> i & 1;
        if(max_id[tr[p][!v]] >= limit) p = tr[p][!v];
        else p = tr[p][v];
    }
    return val ^ s[max_id[p]];
}
int main(){
    scanf("%d%d", &n, &m);
    root[0] = ++cur;
    max_id[0] = -1;
    insert(0, 23, 0, root[0]);
    for(int i = 1; i <= n; i++) {
        int x; scanf("%d", &x);
        s[i] = s[i - 1] ^ x;
        root[i] = ++cur;
        insert(i, 23, root[i - 1], root[i]);
    }
    for(int i = 1; i <= m; i++) {
        char op[2];
        scanf("%s", op);
        if(*op == 'A') {
            int x;scanf("%d", &x);
            root[++n] = ++cur;
            s[n] = s[n - 1] ^ x;
            insert(n, 23, root[n - 1], root[n]);
        }
        else {
            int l, r, x; scanf("%d%d%d", &l, &r, &x);
            printf("%d\n", query(root[r - 1], x ^ s[n], l - 1));
        }
    }
    return 0;
}
```

## 主席树（区间第k大的数）
```cpp
//主席树 支持单点修改 如果要区间修改则要树套树（线段树套平衡树）
//树的结构不变 可以持久化
const int N = 1e5 + 10, M = 1e4 + 10;
int a[N];
vector<int> nums;
int root[N], cur;
struct sgementTree{
    int l, r, cnt;//维护l~r中有多少不同的数(记为cnt)
}tr[N * 4 + N * __lg(N)];//单点修改

int find(int a) {return lower_bound(nums.begin(), nums.end(), a) - nums.begin();}

//cnt不要动
int build(int l, int r) {
    int p = ++cur;
    if(l == r) return p;
    int mid = l + r >> 1;
    tr[p].l = build(l, mid), tr[p].r = build(mid + 1, r);
    return p;
}

int insert(int p, int l, int r, int x) {
    int q = ++cur;
    tr[q] = tr[p];
    if(l == r) {
        tr[q].cnt++;
        return q;
    }
    int mid = l + r >> 1;
    if(x <= mid) tr[q].l = insert(tr[p].l, l, mid, x);
    else tr[q].r = insert(tr[p].r, mid + 1, r, x);
    tr[q].cnt = tr[tr[q].l].cnt + tr[tr[q].r].cnt;
    return q;
}

int query(int q, int p, int l, int r, int x) {
    if(l == r) return r;
    int cnt = tr[tr[q].l].cnt - tr[tr[p].l].cnt;
    int mid = l + r >> 1;
    if(x <= cnt) return query(tr[q].l, tr[p].l, l, mid, x);
    else return query(tr[q].r, tr[p].r, mid + 1, r, x - cnt);
}
int main(){
    int n, m; scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i++){
        scanf("%d", &a[i]);
        nums.push_back(a[i]);
    }
    sort(nums.begin(), nums.end());
    nums.erase(unique(nums.begin(), nums.end()), nums.end());
    
    //~前缀和
    root[0] = build(0, nums.size() - 1);
    for(int i = 1; i <= n; i++) {
        root[i] = insert(root[i - 1], 0, nums.size() - 1, find(a[i]));
    }

    while(m--) {
        int l, r, k; 
        scanf("%d%d%d", &l, &r, &k);
        printf("%d\n", nums[query(root[r], root[l - 1], 0, nums.size() - 1, k)]);
    }
    return 0;
}
```

## Treap树
### 功能
1. 插入x
2. 删除x
1. 查询数值x的排名（多个相同的数输出最小排名）
1. 查询排名为x的数值
1. 求数值x的前驱
1. 求数值x的后继
### 注意
* 插入、删除对应set中的`inesrt`和`erase`，指定结点(非数值)的前驱后继为`++`/`--`
* set提供`rbegin()`和`rend()`的迭代器
```cpp
#include<bits/stdc++.h>
using namespace std;

#define _DEBUG
#define fi first
#define se second
#define all(a) a.begin(),a.end()
#ifdef _DEBUG
#define debug(a) cout << #a << '-' << a << endl;
#else
#define debug(a) 
#endif
#define eb empalce_back

typedef long long LL;
typedef pair<int, int> pii;

const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 1e5 + 10;

struct heap{
    int l, r;
    int val, dat;
    int cnt, size;//记录结点副本数（同成绩）和子树大小
}a[N]; //初始化均为0,表示空结点
int cur, root, n;

int New(int val) {
    a[++cur].val = val;
    a[cur].dat = rand();
    a[cur].cnt = a[cur].size = 1;
    return cur;
}

void update(int p) {
    a[p].size = a[a[p].l].size + a[a[p].r].size + a[p].cnt; //包含其本身
}

void build() {
    New(-inf);
    New(inf);
    root = 1, a[1].r = 2;
    update(root);
}

int getRankOfVal(int p, int val) {
    if(p == 0) return 0;//404nofound
    if(val == a[p].val) return a[a[p].l].size + 1;
    if(val < a[p].val) return getRankOfVal(a[p].l, val);
    return getRankOfVal(a[p].r, val) + a[a[p].l].size + a[p].cnt;
}

int getValByRank(int p, int rank) {
    if(p == 0) return inf;
    if(a[a[p].l].size >= rank) return getValByRank(a[p].l, rank);
    if(a[a[p].l].size + a[p].cnt >= rank) return a[p].val;
    return getValByRank(a[p].r, rank - a[a[p].l].size - a[p].cnt);//这波操作神似主席树，但是不一样，size维护子树大小神来之笔
}

void zig(int &p) {
    int q = a[p].l;
    a[p].l = a[q].r, a[q].r = p, p = q;
    update(a[p].r), update(p);//此时pq已经呼唤
}

void zag(int &p) {
    int q = a[p].r;
    a[p].r = a[q].l, a[q].l = p, p = q;
    update(a[p].l), update(p);
}

void insert(int &p, int val) {//&是灵魂
    if(p == 0) {
        p = New(val);
        return;
    }
    if(val == a[p].val) {
        a[p].cnt++;
        update(p);
        return;
    }
    if(val < a[p].val) {
        insert(a[p].l, val);
        if(a[p].dat < a[a[p].l].dat) zig(p);
    }
    else {
        insert(a[p].r, val);
        if(a[p].dat > a[a[p].r].dat) zag(p);
    }
    update(p);//递归的目的就在于此
}

int getPre(int val) {
    int ans = 1; //a[1].val = -inf
    int p = root;
    while(p) {
        if(val == a[p].val) {
            if(a[p].l > 0) {
                p = a[p].l;
                while(a[p].r > 0) //只会转弯一次！
                    p = a[p].r;
                ans = p;
            }
            break;
        }
        if(a[p].val < val && a[p].val > a[ans].val) ans = p;
        p = val < a[p].val ? a[p].l : a[p].r;
    }
    return a[ans].val;
}

int getNext(int val) {
    int ans = 2; //a[2].val = inf;
    int p = root;
    while(p) {
        if(val == a[p].val) {
            if(a[p].r > 0) {
                p = a[p].r;
                while(a[p].l > 0)
                    p = a[p].l;
                ans = p;
            }
            break;
        }
        //分居两端 则ans更新
        //没经过一个节点，都尝试更新后继 直白来看就好 不用深究——如果a[p]比ans小 但是比所求大 则更新
        if(a[p].val > val && a[p].val < a[ans].val) ans = p;
        p = val < a[p].val ? a[p].l : a[p].r;
    }
    return a[ans].val;
}


//思路详见算法竞赛进阶教程
void remove(int &p, int val) {
    if(p == 0) return;
    if(val == a[p].val) {
        if(a[p].cnt > 1) {
            a[p].cnt--;
            update(p);
            return;
        }
        if(a[p].l || a[p].r) {
            if(a[p].r == 0 || a[a[p].l].dat > a[a[p].r].dat) 
                zig(p), remove(a[p].r, val);
            else 
                zag(p), remove(a[p].l, val);
            update(p);
        }
        else p = 0;
        return;
    }
    val < a[p].val ? remove(a[p].l, val) : remove(a[p].r, val);
    update(p);
}

void solve() {
    int x, opt;
    scanf("%d%d", &opt, &x);
    switch(opt) {
        case 1:
            insert(root, x);
            break;
        case 2:
            remove(root, x);
            break;
        case 3:
            printf("%d\n", getRankOfVal(root, x) - 1);
            break;
        case 4:
            printf("%d\n", getValByRank(root, x + 1)); //rank increase by 1?
            break;
        case 5:
            printf("%d\n", getPre(x));
            break;
        case 6:
            printf("%d\n", getNext(x));
            break;
    }
}

int main() {
    build();
    int T = 1; 
    cin >> T;
    while(T--) solve();
    return 0;
}
```

# 树状数组
> 题目最难的就在于看出是树状数组
### 反思
1. 只要涉及到区间求和、单点修改的都可以算做树状数组
> 本题就是求比某个数小的所有数，即`1~n`的前缀和；每次操作完之后加入一个点
2. 妙处：从前往后、从后往前依次遍历，直接相乘得到答案
```cpp
#include <bits/stdc++.h>
using namespace std;
#define LL long long
#define For(i, x, y) for(__typeof(x) i = (x) - ((x) > (y)); i != (y) - ((x) > (y)); (x) > (y) ? i-- : i++)
const int N = 2e5 + 10;
int tr[N], gre[N], low[N];
int q[N];
int n;

inline int lowbit(int x) {
    return x & (-x);
}

inline void add(int x, int c) {
    for(int i = x; i <= n ; i += lowbit(i)) tr[i] += c;
} 

inline int sum(int x) {
    int res = 0;
    for(int i = x; i; i -= lowbit(i)) res += tr[i];
    return res;
}

int main(){
    scanf("%d", &n);
    For(i, 1, n + 1) scanf("%d", &q[i]);//begin,end
    For(i, 1, n + 1) {
        int y = q[i];
        gre[i] = sum(n) - sum(y);
        low[i] = sum(y - 1);
        add(y, 1);  
    }

    memset(tr, 0, sizeof tr);//一鱼两吃

    LL res1 = 0, res2 = 0;
    For(i, n + 1, 1) {//反着求，牛！
        int y = q[i];
        res1 += gre[i] * 1LL * (sum(n) - sum(y));
        res2 += low[i] * 1LL * (sum(y - 1));
        add(y, 1);
    }
    printf("%lld %lld\n", res1, res2);
    return 0;
}
```


## 线段树懒标记基本问题

给定一个长度为 N
 的数列 A
，以及 M
 条指令，每条指令可能是以下两种之一：

C l r d，表示把 A[l],A[l+1],…,A[r]
 都加上 d
。

Q l r，表示询问数列中第 l∼r
 个数的和。
对于每个询问，输出一个整数表示答案。
```cpp
#include <bits/stdc++.h>
using namespace std;
#define LL long long
#define For(i, x, y) for(__typeof(x) i = (x) - ((x) > (y)); i != (y) - ((x) > (y)); (x) > (y) ? i-- : i++)
#define ls u << 1
#define rs u << 1 | 1

const int N = 1e5 + 10;
int q[N], n, m;

struct Node{
    int l, r;
    LL sum, add;
}tr[N << 2];

inline void up(Node &a, Node &b, Node &c) {
    a.sum = b.sum + c.sum;
}

inline void down(Node &a, Node &b, Node &c) {
    if(a.add) {
        b.add += a.add; b.sum += 1LL * (b.r - b.l + 1) * a.add;
        c.add += a.add; c.sum += 1LL * (c.r - c.l + 1) * a.add;
        a.add = 0;//这算。。剪枝？
    }
}

inline void build(int u, int l, int r) {
    if(l == r) {
        tr[u] = {l, l, q[l], 0};
    }
    else {
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(ls, l, mid);
        build(rs, mid + 1, r);
        up(tr[u], tr[ls], tr[rs]);
    }
}

inline void change(int u, int ll, int rr, int l, int r, int d) {
    if(ll >= l && rr <= r) {
        tr[u].add += d;
        tr[u].sum += 1LL * d * (rr - ll + 1);
    }
    else {
        down(tr[u], tr[ls], tr[rs]);
        int mid = ll + rr >> 1;
        if(l <= mid) change(ls, ll, mid, l, r, d);//注意！边界1
        if(r > mid) change(rs, mid + 1, rr, l, r, d);
        up(tr[u], tr[ls], tr[rs]);
    }
}

inline Node sum(int u, int ll, int rr, int l, int r) {
    if(ll >= l && rr <= r) {
        return tr[u];
    }
    down(tr[u], tr[ls], tr[rs]);
    int mid = ll + rr >> 1;
    if(l > mid) return sum(rs, mid + 1, rr, l, r);//注意！边界2
    if(r <= mid) return sum(ls, ll, mid, l, r);
    Node tar, left, right;
    left = sum(ls, ll, mid, l, r);//超出无所谓，只要确保有数字在区间内就可以
    right = sum(rs, mid + 1, rr, l, r);
    up(tar, left, right);
    return tar;
}

int main(){
    scanf("%d%d", &n, &m);
    For(i, 1, n + 1) scanf("%d", &q[i]);
    build(1, 1, n);
    while(m--) {
        char op[2];
        int l, r, d;
        scanf("%s%d%d", op, &l, &r);
        if(*op == 'C') {
            scanf("%d", &d);
            change(1, 1, n, l, r, d);
        }
        else {
            printf("%lld", sum(1, 1, n, l, r).sum);puts("");
        }
    }
    return 0;
}
```


## 最大上升子序列和
最长的上升子序列的和不一定是最大的，比如序列(100,1,2,3)的最大上升子序列和为100，而最长上升子序列为(1,2,3)
```cpp
// 树状数组的运用很多样，可以维护区间最大值
#include<bits/stdc++.h>
using namespace std;

#define fi first
#define se second
#define all(a) a.begin(),a.end()
#ifdef _DEBUG
#define debug(a) cout << #a << '-' << a << endl;
#else
#define debug(a) 
#endif

typedef long long LL;
typedef pair<int, int> pii;

const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 1e3 + 10;
int tr[N], n;

inline int lowbits(int i) {
    return i & (-i);
} 

inline int get(int k) {
    int res = 0;
    for(int i = k; i; i -= lowbits(i)) 
        res = max(res, tr[i]);
    return res;
}

inline void add(int k, int v) {
    for(int i = k; i <= n; i += lowbits(i))
        tr[i] = max(tr[i], v);
}
inline void solve() {
    scanf("%d", &n);
    vector<int> q(n + 1), label(n + 1), f(n + 1);
    for(int i = 1; i <= n; i++){
        scanf("%d", &q[i]);
        label[i] = q[i];
    }
    sort(label.begin(), label.end());
    label.erase(unique(label.begin(), label.end()), label.end());
    int res = 0;
    for(int i = 1; i <= n; i++) {
        int pos = lower_bound(label.begin(), label.end(), q[i]) - label.begin() + 1;
        // f[i] = max(f[i], get(pos - 1) + q[i]);
        // add(pos, q[i]);
        f[i] = get(pos - 1) + q[i];
        res = max(f[i], res);
        add(pos, f[i]);
        debug(i)
        debug(res)
    }
    printf("%d\n", res);
}

int main() {
    int T = 1; 
    // cin >> T;
    while(T--) solve();
    return 0;
}
```

## NEW VER

## AC自动机
```cpp
const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 1e4 + 10, S = 55, M = 1e6 + 10;
int n; 
int tr[N * S][26], cnt[N * S], idx;
char str[M];
int q[N * S], ne[N * S];
void insert() {
    int p = 0;
    for(int i = 0; str[i]; i++) {
        int t = str[i] - 'a';
        if(!tr[p][t]) tr[p][t] = ++idx;
        p = tr[p][t];
    }
    cnt[p]++;
}
void build() {
    int hh = 0, tt = -1;
    for(int i = 0; i < 26; i++) {
        if(tr[0][i])
            q[++tt] = tr[0][i];//存下标
    }
    while(tt >= hh) {
        int t = q[hh++];
        for(int i = 0; i < 26; i++) {
            int &p = tr[t][i];
            if(!p) p = tr[ne[t]][i];
            else {
                ne[p] = tr[ne[t]][i];
                q[++tt] = p;
            }
        }
    }
}
void solve() {
    memset(tr, 0, sizeof tr);
    memset(cnt, 0, sizeof cnt);
    memset(ne, 0, sizeof ne);
    scanf("%d", &n);
    for(int i = 0; i < n; i++) {
        scanf("%s", str);
        insert();
    }
    build();
    scanf("%s", str);
    int res = 0;
    for(int i = 0, j = 0; str[i]; i++) {
        int t = str[i] - 'a';
        j = tr[j][t];
        
        int p = j;
        while(p && cnt[p] != -1) {
            res += cnt[p];
            cnt[p] = -1;
            p = ne[p];
        }

        
    }
    printf("%d\n", res);
}

int main() {
    int T = 1; 
    cin >> T;
    while(T--) solve();
    return 0;
}
```

## DINIC网络流
```cpp
#include<bits/stdc++.h>
using namespace std;
#define debug(a) cout << #a << '=' << a << endl;
typedef long long LL;
int n, m, s, t;
const int N = 250, M = 5010 << 1, inf = 0x3f3f3f3f;
int cnt = 2, h[N], e[M], ne[M], w[M];
void add(int a, int b, int val) {
    e[cnt] = b, ne[cnt] = h[a], w[cnt] = val, h[a] = cnt++;
}
int now[N], dep[N];
int bfs() {
    for(int i = 1; i <= n; i++) {
        dep[i] = inf;
    }
    dep[s] = 0;
    now[s] = h[s];
    queue<int> q; q.push(s);
    while(q.size()) {
        int u = q.front(); q.pop();
        for(int i = h[u]; ~i; i = ne[i]) {
            int node = e[i];
            // debug(node)
            // debug(w[i])
            if(w[i] > 0 && dep[node] == inf) {//
                // debug("push")
                // debug(node)
                q.push(node);
                now[node] = h[node];
                dep[node] = dep[u] + 1;
                if(node == t) return 1;
            }
        }
    }
    return 0;
}
int dfs(int u, LL sum) {
    if(u == t) return sum;
    LL k, flow = 0;
    for(int i = now[u]; ~i && sum > 0; i = ne[i]) {
        now[u] = i;
        int v = e[i];
        if(w[i] > 0 && dep[v] == dep[u] + 1) {
            k = dfs(v, min(sum, (LL)w[i]));
            if(k == 0) dep[v] = inf;
            w[i] -= k;
            w[i ^ 1] += k;
            flow += k;
            sum -= k;
        }
    }
    return flow;
}
int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);
    memset(h, -1, sizeof h);
    cin >> n >> m >> s >> t;
    for(int i = 0; i < m; i++) {
        int a, b, val; cin >> a >> b >> val;
        add(a, b, val);
        add(b, a, 0);
    }
    LL ans = 0;
    while(bfs()){
        ans += dfs(s, inf);
        // debug(ans)
        // cout << ans << endl;
    }
    cout << ans << endl;
    return 0;
}
```

## 状态自动机DP
```cpp
const int N = 1010;

int n, m;
int tr[N][4], dar[N], idx;
int q[N], ne[N];
char str[N];

int f[N][N];

int get(char c)
{
    if (c == 'A') return 0;
    if (c == 'T') return 1;
    if (c == 'G') return 2;
    return 3;
}

void insert()
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int t = get(str[i]);
        if (tr[p][t] == 0) tr[p][t] = ++ idx;
        p = tr[p][t];
    }
    dar[p] = 1;
}

void build()
{
    int hh = 0, tt = -1;
    for (int i = 0; i < 4; i ++ )
        if (tr[0][i])
            q[ ++ tt] = tr[0][i];

    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = 0; i < 4; i ++ )
        {
            int p = tr[t][i];
            if (!p) tr[t][i] = tr[ne[t]][i];
            else
            {
                ne[p] = tr[ne[t]][i];
                q[ ++ tt] = p;
                dar[p] |= dar[ne[p]];
            }
        }
    }
}

int main()
{
    int T = 1;
    while (scanf("%d", &n), n)
    {
        memset(tr, 0, sizeof tr);
        memset(dar, 0, sizeof dar);
        memset(ne, 0, sizeof ne);
        idx = 0;

        for (int i = 0; i < n; i ++ )
        {
            scanf("%s", str);
            insert();
        }

        build();

        scanf("%s", str + 1);
        m = strlen(str + 1);

        memset(f, 0x3f, sizeof f);
        f[0][0] = 0;
        for (int i = 0; i < m; i ++ )
            for (int j = 0; j <= idx; j ++ )
                for (int k = 0; k < 4; k ++ )
                {
                    int t = get(str[i + 1]) != k;
                    int p = tr[j][k];
                    if (!dar[p]) f[i + 1][p] = min(f[i + 1][p], f[i][j] + t);
                }

        int res = 0x3f3f3f3f;
        for (int i = 0; i <= idx; i ++ ) res = min(res, f[m][i]);

        if (res == 0x3f3f3f3f) res = -1;
        printf("Case %d: %d\n", T ++, res);
    }

    return 0;
}
```


## jiangly Lazy  Segment Tree
[原题链接](https://codeforces.com/contest/1881/problem/G)
```cpp
#include <bits/stdc++.h>
 
using i64 = long long;
 
template<class Info, class Tag>
struct LazySegmentTree {
    int n;
    std::vector<Info> info;
    std::vector<Tag> tag;
    LazySegmentTree() : n(0) {}
    LazySegmentTree(int n_, Info v_ = Info()) {
        init(n_, v_);
    }
    template<class T>
    LazySegmentTree(std::vector<T> init_) {
        init(init_);
    }
    void init(int n_, Info v_ = Info()) {
        init(std::vector(n_, v_));
    }
    template<class T>
    void init(std::vector<T> init_) {
        n = init_.size();
        info.assign(4 << std::__lg(n), Info());
        tag.assign(4 << std::__lg(n), Tag());
        std::function<void(int, int, int)> build = [&](int p, int l, int r) {
            if (r - l == 1) {
                info[p] = init_[l];
                return;
            }
            int m = (l + r) / 2;
            build(2 * p, l, m);
            build(2 * p + 1, m, r);
            pull(p);
        };
        build(1, 0, n);
    }
    void pull(int p) {
        info[p] = info[2 * p] + info[2 * p + 1];
    }
    void apply(int p, const Tag &v) {
        info[p].apply(v);
        tag[p].apply(v);
    }
    void push(int p) {
        apply(2 * p, tag[p]);
        apply(2 * p + 1, tag[p]);
        tag[p] = Tag();
    }
    void modify(int p, int l, int r, int x, const Info &v) {
        if (r - l == 1) {
            info[p] = v;
            return;
        }
        int m = (l + r) / 2;
        push(p);
        if (x < m) {
            modify(2 * p, l, m, x, v);
        } else {
            modify(2 * p + 1, m, r, x, v);
        }
        pull(p);
    }
    void modify(int p, const Info &v) {
        modify(1, 0, n, p, v);
    }
    Info rangeQuery(int p, int l, int r, int x, int y) {
        if (l >= y || r <= x) {
            return Info();
        }
        if (l >= x && r <= y) {
            return info[p];
        }
        int m = (l + r) / 2;
        push(p);
        return rangeQuery(2 * p, l, m, x, y) + rangeQuery(2 * p + 1, m, r, x, y);
    }
    Info rangeQuery(int l, int r) {
        return rangeQuery(1, 0, n, l, r);
    }
    void rangeApply(int p, int l, int r, int x, int y, const Tag &v) {
        if (l >= y || r <= x) {
            return;
        }
        if (l >= x && r <= y) {
            apply(p, v);
            return;
        }
        int m = (l + r) / 2;
        push(p);
        rangeApply(2 * p, l, m, x, y, v);
        rangeApply(2 * p + 1, m, r, x, y, v);
        pull(p);
    }
    void rangeApply(int l, int r, const Tag &v) {
        return rangeApply(1, 0, n, l, r, v);
    }
    template<class F>
    int findFirst(int p, int l, int r, int x, int y, F pred) {
        if (l >= y || r <= x || !pred(info[p])) {
            return -1;
        }
        if (r - l == 1) {
            return l;
        }
        int m = (l + r) / 2;
        push(p);
        int res = findFirst(2 * p, l, m, x, y, pred);
        if (res == -1) {
            res = findFirst(2 * p + 1, m, r, x, y, pred);
        }
        return res;
    }
    template<class F>
    int findFirst(int l, int r, F pred) {
        return findFirst(1, 0, n, l, r, pred);
    }
    template<class F>
    int findLast(int p, int l, int r, int x, int y, F pred) {
        if (l >= y || r <= x || !pred(info[p])) {
            return -1;
        }
        if (r - l == 1) {
            return l;
        }
        int m = (l + r) / 2;
        push(p);
        int res = findLast(2 * p + 1, m, r, x, y, pred);
        if (res == -1) {
            res = findLast(2 * p, l, m, x, y, pred);
        }
        return res;
    }
    template<class F>
    int findLast(int l, int r, F pred) {
        return findLast(1, 0, n, l, r, pred);
    }
};
 
struct Tag {
    int add = 0;
    
    void apply(const Tag &t) & {
        add = (add + t.add) % 26;
    }
};
 
struct Info {
    std::array<int, 2> l{-1, -1};
    std::array<int, 2> r{-1, -1};
    bool ok = true;
    
    void apply(const Tag &t) & {
        for (auto &x : l) {
            if (x != -1) {
                x = (x + t.add) % 26;
            }
        }
        for (auto &x : r) {
            if (x != -1) {
                x = (x + t.add) % 26;
            }
        }
    }
};
Info operator+(const Info &a, const Info &b) {
    Info c;
    c.ok = a.ok && b.ok;
    if (a.r[0] == b.l[0] && a.r[0] != -1) {
        c.ok = false;
    }
    if (a.r[0] == b.l[1] && a.r[0] != -1) {
        c.ok = false;
    }
    if (a.r[1] == b.l[0] && a.r[1] != -1) {
        c.ok = false;
    }
    if (a.l[0] == -1) {
        c.l = b.l;
    } else if (a.l[1] == -1) {
        c.l[0] = a.l[0];
        c.l[1] = b.l[0];
    } else {
        c.l = a.l;
    }
    if (b.r[0] == -1) {
        c.r = a.r;
    } else if (b.r[1] == -1) {
        c.r[0] = b.r[0];
        c.r[1] = a.r[0];
    } else {
        c.r = b.r;
    }
    return c;
}
 
void solve() {
    int n, m;
    std::cin >> n >> m;
    
    std::string s;
    std::cin >> s;
    
    std::vector<Info> init(n);
    for (int i = 0; i < n; i++) {
        init[i].l[0] = init[i].r[0] = s[i] - 'a';
    }
    
    LazySegmentTree<Info, Tag> seg(init);
    
    while (m--) {
        int o, l, r;
        std::cin >> o >> l >> r;
        l--;
        
        if (o == 1) {
            int x;
            std::cin >> x;
            seg.rangeApply(l, r, {x});
        } else {
            std::cout << (seg.rangeQuery(l, r).ok ? "YES" : "NO") << "\n";
        }
    }
}
```

## tarjen+拓扑+dp
```cpp
#include <cstdio>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

typedef long long LL;

const int N = 100010, M = 2000010;

int n, m, mod;
int h[N], hs[N], e[M], ne[M], idx;
int dfn[N], low[N], timestamp;
int stk[N], top;
bitsete<N> in_stk;
int id[N], scc_cnt, scc_size[N];
int f[N], g[N];

void add(int h[], int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void tarjan(int u)
{
    dfn[u] = low[u] = ++ timestamp;
    stk[ ++ top] = u, in_stk[u] = true;

    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!dfn[j])
        {
            tarjan(j);
            low[u] = min(low[u], low[j]);
        }
        else if (in_stk[j]) low[u] = min(low[u], dfn[j]);
    }

    if (dfn[u] == low[u])
    {
        ++ scc_cnt;
        int y;
        do {
            y = stk[top -- ];
            in_stk[y] = false;
            id[y] = scc_cnt;
            scc_size[scc_cnt] ++ ;
        } while (y != u);
    }
}

int main()
{
    // ------多组样例需要如下初始化--------
    /*
    memset(dfn, 0, sizeof dfn);
    memset(scc_size, 0, sizeof scc_size);
    memset(g, 0, sizeof g);
    memset(f, 0, sizeof f);
    in_stk.reset();
    idx = 0, timestamp = 0, scc_cnt = 0;
    */
    memset(h, -1, sizeof h);
    memset(hs, -1, sizeof hs);

    scanf("%d%d%d", &n, &m, &mod);
    while (m -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(h, a, b);
    }

    for (int i = 1; i <= n; i ++ )
        if (!dfn[i])
            tarjan(i);

    unordered_set<LL> S;    // (u, v) -> u * 1000000 + v
    for (int i = 1; i <= n; i ++ )
        for (int j = h[i]; ~j; j = ne[j])
        {
            int k = e[j];
            int a = id[i], b = id[k];
            LL hash = a * 1000000ll + b;
            if (a != b && !S.count(hash))
            {
                add(hs, a, b);
                S.insert(hash);
            }
        }

    for (int i = scc_cnt; i; i -- )
    {
        if (!f[i])
        {
            f[i] = scc_size[i];
            g[i] = 1;
        }
        for (int j = hs[i]; ~j; j = ne[j])
        {
            int k = e[j];
            if (f[k] < f[i] + scc_size[k])
            {
                f[k] = f[i] + scc_size[k];
                g[k] = g[i];
            }
            else if (f[k] == f[i] + scc_size[k])
                g[k] = (g[k] + g[i]) % mod;
        }
    }

    int maxf = 0, sum = 0;
    for (int i = 1; i <= scc_cnt; i ++ )
        if (f[i] > maxf)
        {
            maxf = f[i];
            sum = g[i];
        }
        else if (f[i] == maxf) sum = (sum + g[i]) % mod;

    printf("%d\n", maxf);
    printf("%d\n", sum);

    return 0;
}
```

## 有依赖的背包问题
```cpp
const int inf = 0x3f3f3f3f, mod = 1e9 + 7, N = 110;
int dp[N][N];

void solve() {
    #define only_one
    int n, v; cin >> n >> v;
    vector<vector<int>> e(n + 5);
    vector<pii> info(n + 5);
    info[0] = {0, 0};
    for(int i = 1; i <= n; i++) {
        int vi, wi, pi; cin >> vi >> wi >> pi;
        info[i] = {vi, wi};
        if(pi == -1) e[0].eb(i);
        else e[pi].eb(i);
        // debug(i)
    }
    auto dfs = [&](auto self, int u, int f)->void {
        int vv = info[u].fi;
        for(auto c: e[u]) {//枚举组数
            if(c == f) continue;
            self(self, c, u);
            // 以体积的枚举代替物品的选择，便是背包问题的优化。某一个体积表示一大类，一类中好多方案，但是不用管。
            for(int j = v - info[u].fi; j >= 0; j--) {//枚举体积
                for(int k = 0; k <= j; k++) {//枚举决策
                    dp[u][j] = max(dp[u][j], dp[u][j - k] + dp[c][k]);
                }
            }
        }
        for(int i = v; i >= info[u].fi; i--) dp[u][i] = dp[u][i - info[u].fi] + info[u].se;
        for(int i = 0; i < info[u].fi; i++) dp[u][i] = 0;
    };
    dfs(dfs, 0, -1);
    cout << dp[0][v] << '\n';
}
```

## 背包求具体方案
> 拓扑图思想
```cpp
#include<bits/stdc++.h>
using namespace std;

int main() {
    //求解顺序--> 背包顺序无所谓，求路径一定从前往后，因此递推要从后往前
    int n, m; scanf("%d%d", &n, &m);
    vector<int> v(n + 1), w(n + 1);
    for(int i = 1; i <= n; i++) {
        scanf("%d%d", &w[i], &v[i]);
    }
    vector<vector<int>> f(n + 5, vector<int> (m + 5, 0));
    for(int i = n; i; i--) {
        for(int j = 0; j <= m; j++) {
            f[i][j] = f[i + 1][j];
            if(j >= w[i]) f[i][j] = max(f[i][j], f[i + 1][j - w[i]] + v[i]);
        }
    }
    
    int j = m;
    for(int i = 1; i <= n; i++) {
        // for(int j = m; j >= w[i]; j--)
            // cout << f[i][j] << " ;; " << f[i - 1][]
            if(j >= w[i] && f[i][j] == f[i + 1][j - w[i]] + v[i]) {
                cout << i << " ";
                j -= w[i];
                // break;
            }
    }
    return 0;
}
```

## NTT
```cpp
int read() {
  int x = 0, f = 1;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
    if (ch == '-') f = -1;
    ch = getchar();
  }
  while (ch <= '9' && ch >= '0') {
    x = 10 * x + ch - '0';
    ch = getchar();
  }
  return x * f;
}

void print(int x) {
  if (x < 0) putchar('-'), x = -x;
  if (x >= 10) print(x / 10);
  putchar(x % 10 + '0');
}

const int N = 300100, P = 998244353;

int qpow(int x, int y) {
  int res(1);
  while (y) {
    if (y & 1) res = 1ll * res * x % P;
    x = 1ll * x * x % P;
    y >>= 1;
  }
  return res;
}

int r[N];

void ntt(int *x, int lim, int opt) {
  int i, j, k, m, gn, g, tmp;
  for (i = 0; i < lim; ++i)
    if (r[i] < i) swap(x[i], x[r[i]]);
  for (m = 2; m <= lim; m <<= 1) {
    k = m >> 1;
    gn = qpow(3, (P - 1) / m);
    for (i = 0; i < lim; i += m) {
      g = 1;
      for (j = 0; j < k; ++j, g = 1ll * g * gn % P) {
        tmp = 1ll * x[i + j + k] * g % P;
        x[i + j + k] = (x[i + j] - tmp + P) % P;
        x[i + j] = (x[i + j] + tmp) % P;
      }
    }
  }
  if (opt == -1) {
    reverse(x + 1, x + lim);
    int inv = qpow(lim, P - 2);
    for (i = 0; i < lim; ++i) x[i] = 1ll * x[i] * inv % P;
  }
}

int A[N], B[N], C[N];

char a[N], b[N];

int main() {
  int i, lim(1), n;
  scanf("%s", &a);
  n = strlen(a);
  for (i = 0; i < n; ++i) A[i] = a[n - i - 1] - '0';
  while (lim < (n << 1)) lim <<= 1;
  scanf("%s", &b);
  n = strlen(b);
  for (i = 0; i < n; ++i) B[i] = b[n - i - 1] - '0';
  while (lim < (n << 1)) lim <<= 1;
  for (i = 0; i < lim; ++i) r[i] = (i & 1) * (lim >> 1) + (r[i >> 1] >> 1);
  ntt(A, lim, 1);
  ntt(B, lim, 1);
  for (i = 0; i < lim; ++i) C[i] = 1ll * A[i] * B[i] % P;
  ntt(C, lim, -1);
  int len(0);
  for (i = 0; i < lim; ++i) {
    if (C[i] >= 10) len = i + 1, C[i + 1] += C[i] / 10, C[i] %= 10;
    if (C[i]) len = max(len, i);
  }
  while (C[len] >= 10) C[len + 1] += C[len] / 10, C[len] %= 10, len++;
  for (i = len; ~i; --i) putchar(C[i] + '0');
  puts("");
  return 0;
}
```

## 手写堆（操作补）
```cpp
        if (!strcmp(op, "I")) {//插入a
            scanf("%d", &a);
            h[++cnt] = a; //!
            hp[cnt] = ++k;
            ph[k] = cnt;
            up(cnt);
        }
        else if (!strcmp(op, "PM")) printf("%d\n", h[1]);//打印最小值
        else if (!strcmp(op, "DM")) {//删除最小值（唯一）
            heap_swap(cnt, 1);
            cnt --; down(1);
        }
        else if (!strcmp(op, "D")) {//删除第k个插入的值
            scanf("%d", &a);
            int t = ph[a];//
            heap_swap(cnt, ph[a]);
            cnt--; up(t); down(t);
        }
        else {//将第a个插入的值变为b
            scanf("%d%d", &a, &b);
            //int t = ph[a];
            h[ph[a]] = b;
            up(ph[a]); down(ph[a]);
        }
```

# 背包
## 多重背包问题：
    1. 方法一：二进制优化（详见旧板子）
    2. 方法二：单调队列优化

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 2e4 + 10;
int q[N];

int main() {
    int n, m; cin >> n >> m;
    vector<int> f(m + 100, 0);
    
    for(int i = 0; i < n; i++) {
        int v, w, s; cin >> v >> w >> s;
        vector<int> res(f);
        for(int j = 0; j < v; j++) {
            int hh = 0, tt = -1;
            for(int k = j; k <= m; k += v) {
                if(hh <= tt && q[hh] < k - s * v) hh++;
                //维护严格单减的，队头最长
                while(hh <= tt && res[q[tt]] + (k - q[tt]) / v * w <= res[k]) tt--;
                q[++tt] = k;
                f[k] = res[q[hh]] + (k - q[hh]) / v * w;
            }
        }
    }
    
    cout << f[m] << endl;
    return 0;
}
```

## 数位dp
```cpp
#include <bits/stdc++.h>
using namespace std;
const int mod=1e9+7;
int d[33][2][2][2];
int l[33],r[33];
int X,Y;
int dfs(int len,bool l1,bool l2,bool ze){
    if(len==-1)return 1;
    if(d[len][l1][l2][ze]!=-1)return d[len][l1][l2][ze];
    int ma1=(l1?l[len]:1);
    int ma2=(l2?r[len]:1);
    int ans=0;
    for(int i=0;i<=ma1;i++){
        for(int j=0;j<=ma2;j++){
            if(i&j)continue;
            //t是之后每个方案的贡献
            int t=-1;
            if(ze&&(i||j))t=len+1;//当前位是最高位1,log+1
            else t=1;//当前位不是最高位1,那么只需要为最高位求方案数即可.
            ans=(ans+1ll*dfs(len-1,l1&&i==ma1,l2&&j==ma2,ze&&!i&&!j)*t%mod)%mod;
        }
    }
    return d[len][l1][l2][ze]=ans;
}
int solve(){
    memset(d,-1,sizeof d);
    int len=0;
    while(X||Y){
        l[len]=(X&1);
        r[len]=(Y&1);
        X>>=1;
        Y>>=1;
        len++;
    }
    return dfs(len-1,1,1,1);
}
signed main(){
    int T;scanf("%d",&T);
    while(T--){
        scanf("%d%d",&X,&Y);
        int ans=(solve()-1+mod)%mod;//减掉i和j同时为0时候的一个贡献.
        printf("%d\n",ans);
    }
    return 0;
}
```

## 线性dp
1. 其实极为常见。关键在于无后效性
2. 将A变为B至少需要改变多少？可以增加、删除、修改
```cpp
// init()
//a[i] == b[j] --> f[i][j] = f[i - 1][j - 1]
// a[i] != b[j] --> f[i][j] = min(f[i - 1][j - 1] + 1, f[i][j - 1] + 1, f[i - 1][j] + 1);

#include<bits/stdc++.h>
using namespace std;

#define tra(i, x, y) for (int i = x; i <= y; i++)
const int N = 1010;
char a[N], b[N];
int f[N][N];

int main() {
    int n, m; scanf("%d%s%d%s", &n, a + 1, &m, b + 1);
    
    //这个初始化。。。没想到确实，二维数组的初始化不可能一个就够
    tra(i, 1, n) f[i][0] = i;
    tra(j, 1, m) f[0][j] = j;
    tra(i, 1, n) {
        tra(j, 1, m) {
            f[i][j] = min(f[i][j - 1] + 1, f[i - 1][j] + 1);
            if (a[i] == b[j]) f[i][j] = f[i - 1][j - 1];
            else f[i][j] = min(f[i - 1][j - 1] + 1, f[i][j]);
        }
    }
    
    printf("%d\n", f[n][m]);
    return 0;
}
```

## 状态机dp
* 设计密码：不能存在S的子串
```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 55, mod = 1e9 + 7;
int n, m;
char str[N];
int f[N][N];
int main() {
	cin >> n >> (str + 1);
	m = strlen(str + 1);

	int next[N] = {0};
	for(int i = 2, j = 0; i <= n; i++) {
		while(j && str[i] != str[j + 1])
			j = next[j];
		if(str[i] == str[j + 1])
			j++;
		next[i] = j;
	}

	f[0][0] = 1;
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < m; j++) {
			for(char k = 'a'; k <= 'z'; k++) {
				int u = j;
				while(u && k != str[u + 1])
					u = next[u];
				if(k == str[u + 1])
					u++;
				if(u < m)
					f[i + 1][u] = (f[i + 1][u] + f[i][j]) % mod;
			}
		}
	}

	int res = 0;
	for(int i = 0; i < m; i++) 
		res = (res + f[n][i]) % mod;

	cout << res << endl;
	return 0;
}
```

## 权值线段树
> 线段树的理解尚且不够深刻，l~r可以维护一个很大的值域，但是树的节点不用开的很大，因为u顶多就那么一些，从1开始往下细分，所以不用离线处理。
```cpp
#include <iostream>
using namespace std;
typedef long long ll;
const int N = 2e5 + 3;
const int INF = 1e9;
namespace SegmentTree
{
    struct node
    {
        int l, r, num;
        ll sum;
    } st[N << 5];
    int tot;
    void update(int &id, int segl, int segr, int pos, int val)
    {
        if (!id)
            id = ++tot;
        st[id].num += val;
        st[id].sum += val * pos;
        if (segl == segr)
            return;
        int mid = (segl + segr) >> 1;
        if (pos <= mid)
            update(st[id].l, segl, mid, pos, val);
        else
            update(st[id].r, mid + 1, segr, pos, val);
    }
    int query(int id, int segl, int segr, ll sum)
    {
        if (segl == segr)
            return (sum<=0?0:(sum+segl-1)/segl);
        int mid = (segl + segr) >> 1;
        if (st[st[id].r].sum >= sum)
            return query(st[id].r, mid + 1, segr, sum);
        return query(st[id].l, segl, mid, sum - st[st[id].r].sum) + st[st[id].r].num;
    }
};
int a[N];
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n, T, rt = 0, cnt = 0;
    ll sum = 0;
    cin >> n >> T;
    for (int i = 1; i <= n; ++i)
    {
        cin >> a[i];
        if (a[i] > 0)
            SegmentTree::update(rt, 1, INF, a[i], 1), ++cnt;
        sum += a[i];
    }
    for (int kase = 1; kase <= T; ++kase)
    {
        int x, v;
        cin >> x >> v;
        if (a[x] > 0)
            SegmentTree::update(rt, 1, INF, a[x], -1), --cnt;
        if (v > 0)
            SegmentTree::update(rt, 1, INF, v, 1), ++cnt;
        sum = sum - a[x] + v;
        a[x] = v;
        cout << cnt - SegmentTree::query(1, 1, INF, sum) +1<< "\n";
    }
    return 0;
}
```

## 最大区间和
```cpp
signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    int n; cin >> n;
    vector<int> arr(n + 1, 0);
    for(int i = 1; i <= n; i++) cin >> arr[i];

    vector<int> f(n + 1, 0);
    vector<int> pos(n + 1, 0);
    for(int i = 1; i <= n; i++) {
        if(f[i - 1] + arr[i] > arr[i]) {
            f[i] = f[i - 1] + arr[i];
            pos[i] = pos[i - 1];
        }
        else {
            f[i] = arr[i];
            pos[i] = i;
        }
    }
    
    int ans = -inf;
    int l = 0, r = 0;
    for(int i = 1; i <= n; i++) {
        if(f[i] > ans) {
            ans = f[i];
            l = pos[i];
            r = i;
        }
    }

    cout << ans << '\n';

    return 0;
}
```
## 快读
```cpp
inline __int128 read(){
    __int128 x = 0, f = 1;
    char ch = getchar();
    while(ch < '0' || ch > '9'){
        if(ch == '-')
            f = -1;
        ch = getchar();
    }
    while(ch >= '0' && ch <= '9'){
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}
inline void print(__int128 x){
    if(x < 0){
        putchar('-');
        x = -x;
    }
    if(x > 9)
        print(x / 10);
    putchar(x % 10 + '0');
}
int main(void){
    __int128 a = read();
    __int128 b = read();
    print(a + b);
    cout << endl;
    return 0;
}

```

## splay(文艺平衡树)
```cpp
#include<bits/stdc++.h>
using namespace std;

#define debug(a) cout << #a << " = " << a << endl;

const int N = 1e5 + 10;
// fa: father; ch: the number of its left/right child; val: value; cnt: number of this value; sz: size of the subtree rooted it.
int rt, tot, fa[N], ch[N][2], val[N], cnt[N], sz[N], lazy[N];
int a[N], l, r;
int n, m;
struct Splay{
    void maintain(int x) {
        sz[x] = sz[ch[x][0]] + sz[ch[x][1]] + cnt[x];
    }

    // return 0 if its father's left child, 1 if right.
    bool get(int x) {
        return x == ch[fa[x]][1];
    }

    void clear(int x) {
        ch[x][0] = ch[x][1] = fa[x] = val[x] = sz[x] = cnt[x] = lazy[x] = 0;
    }

    void rotate(int x) {
        int y = fa[x], z = fa[y], chk = get(x);
        ch[y][chk] = ch[x][chk ^ 1];
        if (ch[x][chk ^ 1]) fa[ch[x][chk ^ 1]] = y;


        ch[x][chk ^ 1] = y;
        fa[y] = x;

        fa[x] = z;
        if(z) ch[z][y == ch[z][1]] = x;
        
        // order!
        maintain(y);
        maintain(x);
    }
#if(0)
    void splay(int x){
        for(int f = fa[x]; f = fa[x], f; rotate(x)) 
            if(fa[f]) rotate(get(x) == get(f) ? f : x);
        rt = x;
    }
#else
    void splay(int x, int goal = 0) {
        if(goal == 0) rt = x;
        while(fa[x] != goal) {
            int f = fa[x], g = fa[fa[x]];
            if(g != goal) {
                if(get(x) == get(f)) {
                    rotate(f);
                }
                else 
                    rotate(x);
            }
            rotate(x);
        }
    }
#endif

    void ins(int k) {
        if(!rt) {
            val[tot++] = k;
            cnt[tot] ++;
            rt = tot;
            maintain(rt);
        }
        int cur = rt, f = 0;
        while(1) {
            if(val[cur] == k) {
                cnt[cur]++;
                maintain(cur);
                maintain(f);
                splay(cur);
                break;
            }
            f = cur;
            cur = ch[cur][val[cur] < k];
            if(!cur) {
                val[++tot] = k;
                cnt[tot] ++;
                fa[tot] = f;
                ch[f][val[f] < k] = tot;
                maintain(tot);
                maintain(f);
                splay(tot);
                break;
            }
        }
    }
    void targrev(int x) {
        swap(ch[x][0], ch[x][1]);
        lazy[x] ^= 1;
    }
    void pushdown(int x) {
        if(lazy[x]) {
            targrev(ch[x][0]);
            targrev(ch[x][1]);
            lazy[x] = 0;
        }
    }
    int rk(int x) {
        int res = 0, cur = rt;
        while(1) {
            if(x < val[cur]) {
                cur = ch[cur][0];
            }
            else {
                res += sz[ch[cur][0]];
                if(x == val[cur]) {
                    splay(cur);
                    return res + 1;
                }
                res += cnt[cur];
                cur = ch[cur][1];
            }
        }
    }

    //the node of the value ranked No.k
    int kth(int k) {
        int cur = rt;
        while(1) {
            pushdown(cur);
            if(ch[cur][0] && k <= sz[ch[cur][0]]) {
                cur = ch[cur][0];
            } else {
                k -= cnt[cur] + sz[ch[cur][0]];
                if(k <= 0) {
                    splay(cur);
                    return cur;
                }
                cur = ch[cur][1];
            }
        }
    }

    int pre() {
        int cur = ch[rt][0];
        if(!cur) return cur;
        while(ch[cur][1]) cur = ch[cur][1];
        splay(cur);
        return cur;
    }

    int nxt() {
        int cur = ch[rt][1];
        if(!cur) return cur;
        while(ch[cur][0]) cur = ch[cur][0];
        splay(cur);
        return cur;
    }

    // del one node with value k
    void del(int k){
        rk(k); // k becomes the root
        if(cnt[rt] > 1) {
            cnt[rt]--;
            maintain(rt);
            return ;
        }
        if(!ch[rt][0] && !ch[rt][1]) {
            clear(rt);
            rt = 0;
            return;
        }
        if(!ch[rt][0]) {
            int cur = rt;
            rt = ch[rt][1];
            fa[rt] = 0;
            clear(cur);
            return;
        }
        if(!ch[rt][1]) {
            int cur = rt;
            rt = ch[rt][0];
            fa[rt] = 0;
            clear(cur);
            return;
        }
        int cur = rt;
        int x = pre(); // move x to the root and guarantee that the original rt has no left substree!
        fa[ch[cur][1]] = x;
        ch[x][1] = ch[cur][1];
        clear(cur);
        maintain(rt);
    }

    // reverse value between [l, r]
    void reverse(int l, int r) {
        int ll = kth(l), rr = kth(r + 2);
        splay(ll), splay(rr, ll);
        targrev(ch[ch[rt][1]][0]);
    }

    void print(int x) {
        pushdown(x);
        if (ch[x][0]) print(ch[x][0]);
        if (val[x] >= 1 && val[x] <= n) printf("%d ", val[x]);
        if (ch[x][1]) print(ch[x][1]);
    }

  int build(int l, int r, int f) {
    if (l > r) return 0;
    int mid = (l + r) / 2, cur = ++tot;
    val[cur] = a[mid], fa[cur] = f;
    ch[cur][0] = build(l, mid - 1, cur);
    ch[cur][1] = build(mid + 1, r, cur);
    cnt[cur]++;
    maintain(cur);
    return cur;
  }
}tree;


int main() {
    scanf("%d%d", &n, &m);
    for (int i = 0; i <= n + 1; i++) a[i] = i;
    rt = tree.build(0, n + 1, 0);
    while (m--) {
        scanf("%d%d", &l, &r);
        tree.reverse(l, r);
        // tree.print(rt);
    }
    tree.print(rt);
    return 0;
}
```