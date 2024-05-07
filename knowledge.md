## 2023年5月31日
1. set用法
![本地路径](./notepicture/a5e04e1708e95dc99eb9ab4e31d6584.png "set的用法")
    * `set`可以自动排序
    * 注意遍历写法
    * set可以用来存pair，效率和map差不多，为logn

2. 高精度 `long double`
    * double -- 小数点后14/15位；long double --- 后18/19位
    * `0.0l`为0.0自动转为long double
    * 输出 `printf("%.9Lf", (ld)res)`

3. 谨以此dp纪念我的愚蠢，dp真的很玄妙哇
![本地路径](./notepicture/dc8992c19ef6d9fb5a5e0284dd9f256.png)
     > reflect:
     >> 不到万不得已，不要让cin变得麻烦<br>
     >> dp的集合划分一定要非常明晰
 
4. 别人的`read()`函数
```cpp
inline int read(){
    int x=0,w=1;
    char ch=getchar();
    for(;ch>'9'||ch<'0';ch=getchar()) if(ch=='-') w=-1;
    for(;ch>='0'&&ch<='9';ch=getchar()) x=x*10+ch-'0';
    return x*w;
}
```
<br>

5. `unordered_map`不能排序？试试`map`！
![本地路径](./notepicture/mappppp.png)
    * 自动按key排序

1. memset技巧
    * 可以初始化pii
    * 为防止T，`memset(f, 0, sizeof(int) * (n + 1))`


## 2023年6月2日
1. for的终极宏定义：
    ```cpp
    #define For(i, begin, end) for (__typeof(begin) i = (begin) - ((begin) > (end)); i != (end) - ((begin) > (end)); (begin) > (end) ? i-- : i++)
    ```
2. 结构体中的运算符重载

3. 即使数据没有爆long long, `res`只要涉及+/*就要注意开`long long`
eg：
    ```cpp
    long long res += 1ll * bi * cnt;
    ```
4. qsort常用写法：记录下标转移
    ```cpp
    void qsort(int l, int r) {
    if(l >= r) return;
    int ll = l - 1, rr = r + 1;
    int mid = a[ll + rr + 1 >> 1];
    while(ll < rr) {
        while(a[++ll] < mid);
        while(a[--rr] > mid);
        if(ll < rr) {
            swap(a[ll], a[rr]);
            swap(index[ll], index[rr]);
        }
    }
    qsort(ll, r);
    qsort(l, ll - 1);
    }
    ```

## 2023年6月4日
1. 关于相互映射的复习
    ```cpp
    //在swap过程中保持相互映射
    //pionter->heap;;heap->pionter
    void heap_swap(int i, int j) {
    swap(ph[hp[i]], ph[hp[j]]);
    swap(hp[i], hp[j]);
    swap(h[i], h[j]);
    }
    ```
2. `upper_bound`的用法
    ```cpp
    int cnt = a.size() - (upper_bound(a.begin(), a.end(), b[i]) - a.begin());
    ```
    > cnt == a中严格比b[i]大的有几个
    * a is sorted(所需元素左右两边严格比这个元素大/小即可)
    * ` - a.begin()` 可以保证返回 __下标+1__,即去除iterator
    * `lower_bound` 则是大于等于b[i]的

## week 1(?) 2023年6月5日~
### codeforces Rounde877
1. 代码开头模板
    ```cpp
    #include<bits/stdc++.h>
    using namespace std;

    #define For(i, begin, end) for(__typeof(begin) i = (begin) - (begin > end); i != (end) - ((begin) > (end)); (begin) > (end) ? i-- : i ++)
    #define den(a) cout << #a << " = " << a << "\n";
    #define deg(a) cout << #a << " = " << a << " ";
    
    typedef long long LL;
    typedef pair<int, int> pii;
    typedef pair<ll, ll> pll;

    const int mod = 1e9 + 7, INF = 0x3f3f3f3f;

    inline void solve() {
        //init
        //input

        //logic

    }
    int main() {
        int T = 1;
        scanf("%d", &T);
        while(T--) {
            solve();
        }
        return 0;
    }
    ```

2. Reflection on cf873 B2
    > first and foremost, DON'T TRY ANY HARDER THAN DIV1 A in short time

    ## 易错点
    1. `不开long long见祖宗` 在处理 __任何__ 数据时，乘法加法等，注意开 `long long`
    2. 边界问题处理清楚是要一定经验的：不要写屎，你会后悔的。

    ## 反思
    1. __stack单调栈__ 
        为得到左边第一个比它小的数
        * 赋值栈顶——每次用i赋值栈顶，从后往前，维护一个单减的序列，每次栈顶赋值，得到答案
    2. stack存用下标，需要indices所对应的值的时候`q[stk[indices]]`即可
    3. 多多接受题目暗示,eg. `sort[q[a] ~ q[b]]`的成本为 b - a，代表少排一个字母，即明确一个值在哪次排序中可以不需要排序，即`k的左边max < k 的右边min`<br>
    为实现这一效果，并且防止边界问题———— __遍历所有i，找到左边第一个比q[i]小的数(k),和右边第一个比q[i]小的数(x), 加之k左边第一个比他大的数(y),能减免sort的次数为(k - y) * (x - i)__ $ 想要证明? 自己写写看 $ 
> Try to calculate the contribution of each position.
    >> What an important strategy!


## week2 6.12~
### 2023年6月12日
1. 数字、字母的二进制表示
    1. 数字可以拆解为 `k1 * 2 ^ 0 + k2 * 2 ^ 1 + k3 * 2 ^ 2 + ...... + o(余项)` k1, k2, k3...可取0 or 1
    2. 字母：`a == 00001, b == 00010, c == 00011, ......` 2 ^ 5内可表示所有26个字母
2. 想要一个自动排序其中元素的数据结构？
    > 有重复元素用multi_set<br>
    > 无重复元素用set<br>
    * `*(--sett.end())`才是最后一个元素，复杂度logn，底层红黑树
3. 慎用unordered_map(查找可能被卡), 可用map代替

### 2023年6月17日
## cfedu150
1. map的元素调用：
    ```cpp
    for(auto i = mapp.end() - 1; i != mapp.begin() - 1; i--) {
        TYPE a = i->first, b = i->second;
    }
    ```
2. 关于`__builtin__popcount()`：
    * __返回二进制表示中含1的个数__
    * count + `l` or `ll` 分别针对long 和 long long
    * 底层实现：
        > 方法一:将相邻两位相加，可以实现用二进制来表示输入数据中‘1’的个数。然后依次将上半部分和下半部分相和并实现计数。
        ```cpp
        unsigned popcount (unsigned u){
        u = (u & 0x55555555) + ((u >> 1) & 0x55555555);//u的个位十位的二进制就存好了
        u = (u & 0x33333333) + ((u >> 2) & 0x33333333);
        u = (u & 0x0F0F0F0F) + ((u >> 4) & 0x0F0F0F0F);
        u = (u & 0x00FF00FF) + ((u >> 8) & 0x00FF00FF);
        u = (u & 0x0000FFFF) + ((u >> 16) & 0x0000FFFF);
        }
        ```
        > 方法二：哈希暴力<br>
        每`2^8`一次左移，一共256种情况对应所有二进制1的数量

    * reflect：打表的智慧，一般在数据量有限，问询较多的时候适用
3. `assert(/*condition*/)`条件正确时顺利通过；条件错误时报错终止程序，可用于debug
4. B题dp
    ```cpp
    For(j, 0, 5) {
        For(i, 0, 2) {
            For(k, 0, 5) {
                int num = max(j, k);
                int judge = i + (x != k);
                if(judge < 2) dp[1][num][judge] = max(dp[1][num][judge], dp[0][j][i] + (j > x ? -nn[x] : nn[x]));
            }
        }
    }
    ```

### 2023年6月18日
1. `__gcd()`是内置的最大公约数函数，与手写`gcd()`相同，只能对 __正整数__ 使用，因此注意特判
2. ```cpp
    #ifdef _IO
    freopen("C:\\data\\inluogu.txt","r", stdin);
    freopen("C:\\data\\outluogu.txt","w", stdout);//覆盖
    #endif
    /*code*/
    #ifdef _IO
    fclose(stdin);
    fclose(stdout);
    #endif
    ```
3. 邻接表存图注意数组开大两倍

## cf880
> 表现非常差！凡是关于数论的题你都摸不着头脑
> 酱紫爆零？明天打个翻身仗！
1. A题
    + 毫无难度，一个map就能解决的事情，代码写的太慢了---代码能力
    + 对题目的解读毫无经验，样例给了边界情况，你还写了个错屎---思维能力
    + 迭代中维护前一个答案的写法慢吞吞，if的条件判断还在犹豫，边界情况不加以排除特判---corner cases
2. B题
    > 这位更是重量级
    + 本身就是最简单的贪心策略：
        所有元素省出最多的钱，发现总量不够，i.e.`(left = used % g) != 0`<br>
        又，`left < g`显然，则针对一个处理即可
    + 比赛的时候你不知道在瞎推导什么？写了两张纸，p用没有
3. C题，抽空计时做
    + 有思路之后很好写，10min code + 10min debug == AC
    + 但其实思路也很好想，暴力枚举就可以了，n的范围也只有6
    + 为什么自己想的时候会写屎呢？
        + 没有接受题目暗示（数据范围）
        + 执着于分情况讨论，随之而来的`if`特判和`边界问题`
### 反思
1. 并非一无是处吧，至少你慢慢学着数学推导了（虽然这次力气完全用错了地方
    > 先想贪心，再数学证明
2. 多多写代码吧，没有思维也没有码力，你怎么打ACM
3. 多多熟练模板，抓住本质，融会贯通！

### 2023年6月20日
1. 初始化---`()`;赋值---`{}`
2. `struct`：

    ```cpp
    //表示struct Node类型的数组tr[N]
    struct Node {int a, int b, int c} tr[N];
    ```


3. 预处理log2[n]
    ```cpp
    log2[1] = 0;
    for(int i = 2; i <= N; i++) 
        lg2[i] = lg2[i >> 1] + 1;
    ```
## cf881div3
> 还可以，但是小bug让我错了两道题
1. A题
    * 你考虑需不需要特判，需要 __逻辑上__ 和 __运行上__ 都可行，不要脑袋不动，因为数组越界他也不会报错
        eg. 你本来数组为空，你调`d[0]`显然就不对
    * 用`For(i, x, y)`的宏定义时注意`x > y`是不是你想要的反向遍历，若否，则 __特判__
2. B题
    * `<<`和 `>>`千万别反
    * runtime error 先看数组大小

### 2023年6月24日
1. 回文字符串写屎，原因--- `reverse()`与`string的拼接` 使用不熟练

### 2023年6月25日
1. 关于二分边界
    > 私以为 __不要死守什么左闭右开__，有的题没法左闭右开的
    1.  首先，别死循环，两种方法（核心：__r和l都要过火__）
    ```cpp
    while(l < r) {
        int mid = l + r >> 1;//下取整
        if(check()) r = mid;//r过火
        else l = mid + 1;//l过火
    }
    return l;
    ```
    ```cpp
    while(l < r) {
        int mid = l + r + 1 >> 1;//上取整
        if(check()) r = mid - 1;
        else l = mid;
    }
    return l;
    ```
    > 注意，while的判断条件不能变！
    2.  接着，根据题目条件调整l和r的初始值
    ```cpp
    // n == ve.size()
    int l = 0, r = n;//左侧永远取不到,若l = n则判断不存在
    int l = 0, r = n - 1;//左右都去得到，一定得出结果
    int l = 1, r = n;//非下标的角标写法
    ```
2. 什么时候二分？
    1. 所求点的左边一律比它小，右边一律比他大
    2. __对操作进行二分__：操作满足之后，（数组）一律满足某个条件；操作数未达标，（数组）绝对不会满足条件。——用于求第一个满足条件的操作是第几个？[cf881E](https://codeforces.com/contest/1843/problem/E)
3. 善用`vector<int> num(n, 0);`来开数组，免得每次memset

## 2023年6月27日
1. 多个数求`max``min`的语法糖：`maxx = max({a, b, c, ..., });`

## 2023年7月7日
1. 树状数组可以解决的问题：
    * 显性：区间求和、区间修改，均在$O(nlogn)$中
    * __隐形__ ： 动态维护平衡树问题（谜一样的牛、楼兰图腾）
2. `set`没有`sett.begin() + i`的迭代器得到第i大；只有`lower_bound`,`sett.begin() ++`这类
3. 大胆猜测小心求证。注重题目的 __细节与暗示__ ，往往要先化简一步再计算（剪枝）
1. `dp`本质上就是有效枚举，但不穷举。要仔细考虑所有可能情况，有时看似很少、其实很多，有时则是 __反之__


## 2023年7月18日
### cf884
1. 大胆猜想，队友求证
2. 最开始的方向如果有偏差，那么后面就是积重难返。这要靠经验（div1. A）
    > 就他为什么是奇偶呢？相邻两个数绝对不会并到一起，间隔一个的一定能碰到一起，而且还能任意舍弃。
    * 一句话： __透过现象看本质__ 这是ACM的核心技能
3. 构造题目更是如此了，虽然是div1.B，但是规律掌握了还是能A的[CF884D](https://codeforces.com/contest/1844/problem/D)
> 思维题，思路最关键，基本用不上什么算法知识
### cf885
> worst contest ever, but suit for me
1. A题连蒙带猜，很有ACM的风格；B题写屎改了一会儿，各种corner也都改掉了
2. C题mathy，没写出来：
    * 写两个小时需要休息一会儿
    * 数字推导能力
    * 如果corner使得你要维护的东西非常多而且杂乱，基本就要换算法了
### dfs反思[ACWING890](https://www.acwing.com/problem/content/description/892/)
1. 枚举所有情况，基础的dfs画树肯定画的出来，一般会将下标从第一个枚举到最后一个
2. 多写，多练为王道
### 哈希Hash
1. 匹配相同事物的时候用以降低复杂度，满足某个较弱条件即可被分到一组，一组中再去判断[ACWING137](https://www.acwing.com/problem/content/139/)
2. 字符串哈希用以匹配回文串[ACWING139](https://www.acwing.com/problem/content/141/)
    * 回文串奇偶讨论尽量避免
    * 善于用二分解决：有 __单调性__ 的问题，包括答案长度，只要存在一个边界，比它小的均可，比他大的均不可。
### 
1. 基数排序的发明者真是个天才！配合桶排序的根基，竟能如此优雅!详见模板

## 2023年8月1日
> 最近调整心态摆烂，刷题明显下降，无所谓，休息很重要
1. 滑动窗口问题：求满足某条件的最**的子区间
* 需要点数学推导能力,eg出现重复的一定是新加进去的那个点（考虑极限情况，并加以类似贪心的证明）[LC03](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
* template
    ```cpp
    for(int i = 0, j = 0; i < size; i++) {
        //code
        for(; j < i && check(i ~ j); j++) //code
        if(check(i ~ j)) res = max(res, **);
        //code
    }
    ```
2. 关于怎么MOD？*** 针对爆LL的 ***
    * 凡表达式超过一步，则每一步都要mod，最后mod之前要加上mod
    ```cpp
    int ans = (res % mod - n % mod + mod) %= mod;
    ```

## 2023年8月8日
1. 当一道数学题涉及到求根公式，最好封装一个函数来算，避免在`slove()`中写屎
2. 关于一些细微的STL函数，复杂度均为$O(n)$
    ```cpp
    int a[] = {1, 2, 3, 4, 5};
    int minValue = *min_element(a, a + 5); //返回最大值指针
    int maxValue = *max_element(a, a + 5); //返回最小值指针
    int sumValue = accumulate(a, a + 5, 0); //a[i] for i from 1 to n - 1
    ```
3. `umap`的增删复杂度$O(1)$ ,但是`hash`最好还是用数组最快
4. 字符串问题，可以用添加`#`的方式防止奇偶讨论([ACWING139. 回文子串的最大长度](https://www.acwing.com/problem/content/141/)), 或者分割合并成一串的多个字符串(XY_cpp的后缀自动机)
5. 结构体不等号重载
    ```cpp
    bool operator < (const node & x) const{
        if(a == x. a) {
            if(b == x.b) return c < x.c;
            return b < x.b;
        }
        return a < x.a;
    }
    ```
    >  * 还是比较麻烦的，所有能用pii排序则优先用它
    >  * 或者写个`bool cmp(node & a, node & b)`

## 关于USD(disjoint set union 并查集)
* father一定要做成全局变量，而且`slove()`开头不必memset
## 关于概念(以最小生成树MST minimum spanning tree为例)
* 计算机概念庞杂复杂，打算法的就是要了解清楚每一个概念及其相关性质，利用性质做题可能会简单很多，MST是如此，完全二叉树也是如此。
    - eg. `MST`的重要概念
    ```cpp
    设
        连通网G = (V, E)，
        U是V的非空真子集，
        边(u, v)是 所有一端在U中，另一端在V-U的边 中，代价最小的
    则
        在G的所有最小生成树中，一定有一棵包含(u, v) 
    ```
    - 反之，则可得到，`若有U中有其他连向v的边，必然比 w(u, v) 要大，反之亦然`
        > [cf1857G](https://codeforces.com/contest/1857/problem/G)

## 2023年8月11日
1. 离散化
    1. `unique(a, a + n)` 无需排序，作用：去重。
    2. __真正的__ 的离散化——一步到位
        ```cpp
        vector<int> q;//to be discreted
        vector<int> p(q);
        sort(p.begin(), p.end());
        for(int i = 0; i < q.size(); i++) {
            q[i] = lower_bound(p.begin(), p.end(), q[i]) - p.begin();
        }
        ```
        > 复杂度$O(nlogn)$

## 2023年8月15日
1. 积性函数都可以用 __欧拉筛求解__ (详见线性筛素数的代码)
2. 欧拉筛的重要作用是 __欧拉定理__

# CF887 Div1 is God like! More and more!

## 2023年8月28日
1. 第n位取反 `target >> (n + 1) ^= 1`
2. 关于const
    ```cpp
    const char* s = "0000"; 
    s = "1234"; //成立，
    char* const s = "0000";
    s = "1234";//不成立
    ```
3.  关于`stringstream`
    因为传入参数和目标对象的类型会被自动推导出来，所以不存在错误的格式化符的问题
    ```cpp
    int h[N];
    string line;
    getline(cin, line);//input as string
    stringstream ssin(line);//convert into ssin
    while (ssin >> h[n]) n ++ ;//read int from ssin and put into h, split with space
    ```
    - 由下可见，stringstream可用于类型转换
    
    ```cpp
    std::stringstream strstr;
    strstr << "100" << ' ' << "200";
    int a, b;
    ss >> a >> b;
    cout << a << b << endl;
    ```

## 2023年9月9日
1. dp无后效性：并非指后面的选择不会影响前面的选择，而是后面的 __选择与否__ 不会影响前面结点是否能选
    > dp会枚举前面的东西选或不选的所有情况，保留其中的dominance strategy, 供后面随便选择
    * skills：
        - `f[i][j] = max(f[i][j], ...)` often occurs
        - 能否滚动在于新点是否覆盖旧点。如果会，则不行。
2. ```cpp
    merge(all(x.a), all(y.a), back_inserter(z.a));
    //Merges two sorted ranges [first1, last1) and [first2, last2) into one sorted range beginning at d_first.
    //back_inserter() is the last pointer of z.a
    ```
    brilliant DP problem from CF887 D2F

## 2023年9月13日
1. 空间计算(trie树、dp等) ：
    1. int==4字节；longlong=8字节（byte）
    2. 维数相乘*sizeof(int) * n / 1e6 == ans (MB)
    > 如果卡空间，则能用多少用多少，eg. 256MB 开成250MB
2. 求一个数所有的约数（变形）
    ```cpp
    for(int i = 1; i <= n; i++){
        for(int j = i; j <= n; j += i) 
                fenjie[j].eb(i);//logn
    }
    ```
    > 如上求得了$[1, n]$中所有数的所有约数，复杂度为logn
    > 对于每个数字j，遍历其约数的复杂度也为logn

## 2023年11月30日
1. 计时
    ```cpp
    clock_t start, end;
    start = clock();
    func();
    end = clock();
    cout <<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<endl;
    ```

# 备战蓝桥杯
## 2023年12月19日
1. dijkstra：
    * 对每一点为核心只松弛一次，因为后面在堆中的 __要么__ 是`dis`大于它，__要么__ 就是还没有入堆，即是靠着这个点以及后面的点扩展到的，`dis`必然比他大
    * 而对于入堆之前的点只需要判断`dis`，__无需__ 做是否在堆中的判断，因为不同点的松弛很可能更新他的最短距离，使得堆中这个点出现多次。

```
我要极其兴奋地写下以下这段话，谨记AC的心情：
    DEBUG了许久，屎山越堆越高，但是我已经尽量保证简洁的代码了。慢慢就没自信了。一个又一个的bug找到了，我反而越来越不自信了，感觉这份代码有点问题。这是每个字母都是我手打出来的，没有一个字符是来自他人的。也就是没有任何一个人能可靠地保证我的路子是对的。但是，1A。我说，ACM能培养自信力：相信你写出来的东西能毫发无爽地读入、判断、输出。你都不知道他的测试内容是什么，对你来说样例完全透明。你不知道在Running之后出现WA了你该怎么办。你将要陷入各种corner cases的艰难讨论中。从质疑，到确定，再质疑，再确认。在与自己思想的左右互搏中，在否定之否定中，慢慢接近题目的真相。说实话，我非常敬佩出题人，首先得有idea，之后还要造样例，精心设计卡掉所有假做法，设计很多corner cases。作为做题人，我无法想象这需要多么严谨的心思。这是由你一手搭起的大厦，没人会和你的代码一样。而这份东西，ACCEPTED，毋庸置疑就是你的智慧的绝对认可。尤其是瞄了眼题解，之后和他的做法不一样，你是选择看答案，还是相信自己继续写呢？相信自己，是由代价的，时间的代价，也是有收获的，自己的收获。这是培养自信力的过程。
    我很高兴，我能打ACM。自己努力相信自己，经过艰难的DEBUG之后一发入魂，这样的事谁不上瘾？？
    当然，还有沉淀等等的教训，以后再说。
```

## 2023年12月23日
1. int 下需要mod的加与乘
```cpp
int add(int x, int y)
{
	return (((x + y) % MOD) + MOD) % MOD;
}
 
int mul(int x, int y)
{
	return (x * 1ll * y) % MOD;
}
```

2. 最小值
```cpp
memset(a, 207, sizeof a);
```

3. 线段树DEBUG
```cpp
inline void print(int u, int l, int r) {
    if(l == r) printf("%d== %d ;", l, tr[u].sum[0]);
    else {
        int mid = l + r >> 1;
        print(ls, l, mid);
        print(rs, mid + 1, r);
    }
}
```
4. 对拍生成树
```cpp
int dsu[1000005];//1e6
int mapp[10000005];

void create(){
    int n=random(1,10);
    for(int i=n;i>=2;--i){
        dsu[i]=random(1,i-1);//生成一颗以1为根节点的树
    }
    //以上为以1为根，随机还要重新编号对应一下，可以借助随机乱序函数
    for(int i=1;i<=n;++i) mapp[i]=i;
    random_shuffle(mapp+1,mapp+n+1);//随机排列  
    //打印树，必要可以随机边权
    printf("%d\n",n);
    for(int i=2;i<=n;++i){
        printf("%d %d\n",mapp[i],mapp[ dsu[i] ]);
    }
    return;
}
```