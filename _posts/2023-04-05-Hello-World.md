---
title: Hello World
date: 2023-04-05
categories: [Test]
tags: [test]     # TAG names should always be lowercase
math: true
---

## 01背包问题

有 n 件物品和一个容量为 V 的背包，第 i 件物品的体积是 `v[i]` ，价值是`w[i]`  ，求将哪些物品装入背包可使价值总和最大。

01背包问题是最基础的背包问题，特点是：每种物品仅有一件，可以选择放或不放。

如果用暴力解法，即回溯法，思路为暴力枚举每个物品放和不放2种情况。时间复杂度为指数级别，进而才需要使用动态规划进行优化。

回溯法的代码为：

```java
public class Backpack {
    int ans = 0;

    public int backpack(int[] weight, int[] value, int volume) {
        backtrack(weight, value, volume, 0, 0, 0);
        return ans;
    }

    private void backtrack(int[] weight, int[] value, int volume, int curWeight, int curValue, int idx) {
        if (curWeight > volume) return;
        int n = weight.length;
        if (idx >= n) return;
        ans = Math.max(ans, curValue);
        backtrack(weight, value, volume, curWeight, curValue, idx + 1);
        backtrack(weight, value, volume, curWeight + weight[idx], curValue + value[idx], idx + 1);
    }

    @Test
    public void test() {
        int[] weight = {1, 3, 4};
        int[] value = {15, 20, 30};
        int volume = 4;
        System.out.println(backpack(weight, value, volume));
    }
}
```

最标准的做法为动态规划。

第一步：定义DP数组
`dp[i][j]` 表示从下标为`[0...i]`中的物品中选物品，放入一个容量为 j 的背包，可以获得的最大价值。

第二步：确定递推公式
参考回溯的思路，可以从2个方向推出来`dp[i][j]`：

- 不放物品i：此时`dp[i][j] = dp[i-1][j]`
- 放物品i：此时`dp[i][j] = dp[i-1][j-v[i]] + w[i]`

因此状态转移方程为：
$f[i][j] = max(f[i-1][j], f[i-1][j-v[i]] + w[i])$

第三步：DP数组初始化
当背包容量j为0的时候，无法放入任何物品，因此`dp[i][0] = 0`。当物品编号i为0的时候，如果背包能装下物品0，即`j >= v[0]`时，则`dp[0][j] = w[0]`；否则`dp[0][j] = 0`。
因此，初始化的代码为：

```java
for (int j = v[0]; j <= V; j++) {
    dp[0][j] = w[0];
}
```

第四步：确定遍历顺序
有2个遍历的维度：物品（i）与背包容量（j）
事实上，先遍历物品和先遍历背包容量都可以，但是先遍历物品更好理解。

```java
for (int i = 1; i < n; i++) { // 遍历n个物品
    for (int j = 0; j <= V; j++) { // 遍历背包容量
        dp[i][j] = dp[i - 1][j]; // 不装物品i
        if (j >= v[i]) { // 装物品i，需要 j >= v[i]
            dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - v[i]] + w[i]);
        }
    }
}

for (int j = 0; j <= V; j++) { // 遍历背包容量
    for (int i = 1; i < n; i++) { // 遍历n个物品
        dp[i][j] = dp[i - 1][j]; // 不装物品i
        if (j >= v[i]) { // 装物品i，需要 j >= v[i]
            dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - v[i]] + w[i]);
        }
    }
}
```

第五步（可选）：滚动数组优化
从转移方程$f[i][j] = max(f[i-1][j], f[i-1][j-v[i]] + w[i])$可知，第i层的DP数组完全是从第i-1层转移过来的，因此可以把二维数组压缩成一维的，即$f[j] = max(f[j], f[j-v[i]] + w[i])$。

在一维dp数组中，`dp[j]`表示：容量为j的背包，所背的物品价值可以最大为`dp[j]`。

但是需要注意的是，优化之前的二维数组，更新`dp[i][j]`时用的是i-1层的数据，但是优化成一维之后，如果依然是从左到右/从小到大遍历的顺序，那么上一层的数据会被更新的`dp[j]`覆盖掉，因此会产生错误的结果。（即用了第i层的数据去更新第i层，换句话说也就是物品i被放入了不止一次，而倒序遍历是为了保证物品i只被放入背包一次）。因此，在用滚动数组优化时，要注意遍历顺序是背包容量从大到小的。

关于初始化的问题：`dp[j]`表示容量为j的背包，所背的物品价值可以最大为`dp[j]`，那么`dp[0]`就应该是0，因为背包容量为0所背的物品的最大价值就是0。另外，我们在二维数组中习惯先初始化`dp[0][j]`然后遍历时再从`i = 1`开始，但在一位数组中我们可以直接从`i = 0`开始，因为`i = 0`这一轮相当于就是我们在二维数组中的初始化`dp[0][j]`（看下面的代码代入i = 0时就懂了，相当于对于物品0，只要`j>=v[0]`，`dp[j] = w[0]`）。

因此，对于滚动数组优化的结论是：**外层顺序遍历，内层倒序遍历背包**。

```java
for (int i = 0; i < n; i++) { // 遍历n个物品
    for (int j = V; j >= v[i]; j--) { // 遍历背包容量
        dp[j] = Math.max(dp[j], dp[j - v[i]] + w[i]);
    }
}
```

### 模板题：[2. 01背包问题](https://www.acwing.com/problem/content/description/2/)

```java
import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt(); // 物品数量
        int V = sc.nextInt(); // 背包容量
        int[] v = new int[n]; // 每件物品的体积
        int[] w = new int[n]; // 每件物品的价值
        for (int i = 0; i < n; i++) {
            v[i] = sc.nextInt();
            w[i] = sc.nextInt();
        }
        int[][] dp = new int[n][V + 1];
        // 初始化
        for (int j = v[0]; j <= V; j++) {
            dp[0][j] = w[0];
        }
        for (int i = 1; i < n; i++) { // 遍历物品
            for (int j = 0; j <= V; j++) { // 遍历背包容量
                dp[i][j] = dp[i - 1][j]; // if (j < v[i]) 不装
                if (j >= v[i]) { // 装，但需要 j >= v[i]
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - v[i]] + w[i]);
                }
            }
        }
        System.out.println(dp[n - 1][V]);
    }
}
```

### 这个背包能装满吗？[416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/)

题目要求：将数组分割成两个子集，使得两个子集的元素和相等。这句话等价于，如果数组和为`sum`，相当于要求数组里能否出现总和为`sum/2`的子集（当然，如果`sum`为奇数时直接可以返回false）。即背包容量为`sum/2`的01背包问题，如果背包正好装满，则返回true。注：本题中，物品的重量和价值是一样的。

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n < 2) return false;
        int sum = 0, max = 0;
        for (int num : nums) {
            sum += num;
            max = Math.max(max, num);
        }
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        if (max > target) return false;

        int[][] dp = new int[n][target + 1];
        // 初始化
        for (int j = nums[0]; j <= target; j++) {
            dp[0][j] = nums[0];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= nums[i]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - nums[i]] + nums[i]);
                }
            }
        }
        return dp[n - 1][target] == target;
    }
}
```

滚动数组优化：

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if (n < 2) return false;
        int sum = 0, max = 0;
        for (int num : nums) {
            sum += num;
            max = Math.max(max, num);
        }
        if (sum % 2 != 0) return false;
        int target = sum / 2;
        if (max > target) return false;

        int[] dp = new int[target + 1];
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return dp[target] == target;
    }
}
```

### 这个背包最多能装多少？[1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)

题目要求等价于让我们尽可能平均地把石头分成2份，并返回他们的重量之差。由于`target = sum/2`是向下取整，因此较轻的那份的最大重量是`dp[target]`，重量多的那份就是`sum - dp[target]`，因此答案就是`sum - 2 * dp[target]`。

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int n = stones.length;
        int sum = 0;
        for (int x : stones) {
            sum += x;
        }
        int target = sum / 2;
        int[][] dp = new int[n][target + 1];
        for (int j = stones[0]; j <= target; j++) {
            dp[0][j] = stones[0];
        }
        for (int i = 1; i < n; i++) {
            for (int j = 0; j <= target; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= stones[i]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - 1][j - stones[i]] + stones[i]);
                }
            }
        }
        return sum - 2 * dp[n - 1][target];
    }
}
```

滚动数组优化：

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int n = stones.length;
        int sum = 0;
        for (int x : stones) {
            sum += x;
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 0; i < n; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[target];
    }
}
```

### 装满背包有多少种方案？[494. 目标和](https://leetcode.cn/problems/target-sum/description/)

根据题目的要求，并且看到`1 <= nums.length <= 20`这个数据范围，可以尝试用回溯法暴力搜索。

```java
class Solution {
    int ans = 0;

    public int findTargetSumWays(int[] nums, int target) {
        backtrack(nums, target, 0, 0);
        return ans;
    }

    private void backtrack(int[] nums, int target, int sum, int idx) {
        if (idx == nums.length) {
            if (sum == target) ans++;
            return;
        }
        backtrack(nums, target, sum + nums[idx], idx + 1);
        backtrack(nums, target, sum - nums[idx], idx + 1);
    }
}
```

当然，回溯法的时间复杂度是指数级别的，即`O(2^n)`。我们可以尝试用动态规划优化它。

在这个问题中，我们可以把整数数组分为2份，一份为正数数组，一份为负数数组。由于是非负数组，相当于要找把哪些数字变成负数。记负数数组的和为`neg`，由于`(sum - neg) - neg = target`，`neg = (sum - target) / 2`，成立的条件为`sum - target`为非负偶数。

此时的问题就转换为：在数组中选取一些元素，使得这些元素之和等于`neg`，求方案数。即装满容量为`neg`的背包，总共有几种方案？这其实本质上是一个组合问题，和之前求容量为j的背包最多能装多少不同，因此我们也要修改一下DP数组的定义。

我们定义`dp[i][j]` 为：使用前i个物品（i的下标从1开始），填满容量为j的背包，有`dp[i][j]`种方法。同时，我们初始化边界条件`dp[0][0] = 1`，代表不考虑任何数，凑出计算结果为 0 的方案数为 1 种，用于递推。

对于数组`nums`中的第i个元素`num`（i的计数从1开始），基于「不选」和「选」两种决策，我们可以得出：

- 如果`j < num`，则不能选`num`，此时有$dp[i][j] = dp[i - 1][j]$
- 如果`j >= num`
  - 如果不选`num`，方案数为$dp[i][j] = dp[i - 1][j]$
  - 如果选`num`，方案数为$dp[i][j] = dp[i - 1][j] + dp[i - 1][j - num]$

最终的答案为`dp[n][neg]`

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int sum = 0;
        for (int x : nums) {
            sum += x;
        }
        if (target > sum || (sum - target) % 2 != 0) return 0;
        int size = (sum - target) / 2;
        int[][] dp = new int[n + 1][size + 1]; // 注意这里 n + 1
        dp[0][0] = 1; // dp[0][0]为起始条件，代表不考虑任何数，凑出结果为0的方案数有1种
        for (int i = 1; i <= n; i++) {
            int num = nums[i - 1]; // 需要错位
            for (int j = 0; j <= size; j++) {
                dp[i][j] = dp[i - 1][j];
                if (j >= num) {
                    dp[i][j] += dp[i - 1][j - num];
                }
            }
        }
        return dp[n][size];
    }
}
```

滚动数组优化：

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int n = nums.length;
        int sum = 0;
        for (int x : nums) {
            sum += x;
        }
        if (target > sum || (sum - target) % 2 != 0) return 0;
        int size = (sum - target) / 2;
        int[] dp = new int[size + 1];
        dp[0] = 1;
        for (int num : nums) {
            for (int j = size; j >= num; j--) {
                dp[j] += dp[j - num];
            }
        }
        return dp[size];
    }
}
```

总结：求装满背包的方案数

```java
dp[0] = 1;
for (int num : nums) {
    for (int j = size; j >= num; j--) {
        dp[j] += dp[j - num];
    }
}
```

### 装满背包最多用多少个物品？[474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/)

通常与「背包问题」相关的题考察的是 将原问题转换为「背包问题」的能力。要将原问题转换为「背包问题」，往往需要从题目中抽象出「价值」与「成本」的概念。

本题中每个字符串的价值都是1（对答案的贡献都是1），选择的成本是字符串中1的数量和0的数量。因此，本题其实是在问：**在1的数量不超过m，0的数量不超过n的条件下，最大价值是多少**。可以看到这里的背包容量限制从之前的一维提升到了现在的二维。

我们定义：`dp[k][i][j]`代表考虑前k个物品，在数字1容量不超过i，数字0容量不超过j的条件下的最大价值。

状态转移方程：
$dp[k][i][j] = max(dp[k-1][i][j], dp[k-1][i - cnt[k][0]][j - cnt[k][1]] + 1)$
其中`cnt`数组记录了当前字符串中出现的0和1的数量。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][] cnt = new int[len][2];
        for (int i = 0; i < len; i++) {
            String str = strs[i];
            int zero = 0, one = 0;
            for (char c : str.toCharArray()) {
                if (c == '0') zero++;
                else one++;
            }
            cnt[i] = new int[]{zero, one};
        }
        int[][][] dp = new int[len + 1][m + 1][n + 1];
        for (int k = 1; k <= len; k++) {
            int zero = cnt[k - 1][0], one = cnt[k - 1][1];
            for (int i = 0; i <= m; i++) {
                for (int j = 0; j <= n; j++) {
                    int a = dp[k - 1][i][j];
                    int b = (i >= zero && j >= one) ? dp[k - 1][i - zero][j - one] + 1 : 0;
                    dp[k][i][j] = Math.max(a, b);
                }
            }
        }
        return dp[len][m][n];
    }
}
```

一维空间优化：
我们发现：`dp[k][i][j]`不仅仅依赖于上一行，还明确依赖于比i小和比j小的状态。因此我们可以取消掉「物品维度」，然后调整容量的遍历顺序。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int len = strs.length;
        int[][] cnt = new int[len][2];
        for (int i = 0; i < len; i++) {
            int zero = 0, one = 0;
            for (char c : strs[i].toCharArray()) {
                if (c == '0') zero++;
                else one++;
            }
            cnt[i] = new int[]{zero, one};
        }
        int[][] f = new int[m + 1][n + 1];
        for (int k = 0; k < len; k++) {
            int zero = cnt[k][0], one = cnt[k][1];
            for (int i = m; i >= zero; i--) {
                for (int j = n; j >= one; j--) {
                    f[i][j] = Math.max(f[i][j], f[i - zero][j - one] + 1);
                }
            }
        }
        return f[m][n];
    }
}
```

---

## 完全背包问题

完全背包问题和01背包问题唯一的区别是：完全背包问题中，每个物品有无数个，不再有最多1个的限制。

### 模板题：[3. 完全背包问题](https://www.acwing.com/problem/content/3/)

```java
import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int[] v = new int[n + 1];
        int[] w = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            v[i] = sc.nextInt();
            w[i] = sc.nextInt();
        }
        int[][] f = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k * v[i] <= j; k++) {
                    f[i][j] = Math.max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i]);
                }
            }
        }
        System.out.println(f[n][m]);
    }
}
```

### 装满背包有多少种方案？求组合数 [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

根据题意我们很容易判断出来这是一道完全背包问题，因此做以下定义：`dp[i][j]`为考虑前i件物品，凑成总和为j的方案数量。
为了方便初始化，我们让`dp[0][j]`代表不考虑任何物品的情况，并且为了方便递推，我们初始化`dp[0][0] = 1`，其余`dp[0][j] = 0`。代表没有任何硬币的时候，凑成总和为0的方案数为1（仅仅是为了初始化方便递推），凑成其他总和的方案不存在。

对于第i个硬币我们有2种决策：

- 不使用该硬币：`dp[i][j] = dp[i - 1][j]`
- 使用该硬币：由于每个硬币数量无限，因此在容量允许的情况下，方案数量应该是选择「任意个」该硬币的方案总和 `dp[i][j] += dp[i - 1][j - k * coin]`

```java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[][] dp = new int[n + 1][amount + 1];
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            int coin = coins[i - 1];
            for (int j = 0; j <= amount; j++) {
                dp[i][j] = dp[i - 1][j];
                for (int k = 1; k * coin <= j; k++) {
                    dp[i][j] += dp[i - 1][j - k * coin];
                }
            }
        }
        return dp[n][amount];
    }
}
```

时间复杂度：共有`n * amount`个状态需要转移，每个状态转移最多遍历`amount`次。`O(n * amount^2)`

一维数组优化：

- 在二维数组的基础上，直接去掉「物品维度」
- 确保「容量维度」的遍历顺序为从小到大（适用于完全背包）
- 将转移方程改为`dp[j - coin]`，将「容量维度」的遍历起点改为`coin`

关于上面的第二点和第三点的解释：回顾我们在01背包问题的滚动数组优化中提到的，滚动数组中背包容量的遍历顺序必须是从大到小，不然会使用本层更新过的数据覆盖（也就是**物品会被放入背包多次**），而这刚好就是完全背包问题的定义！这也就是为什么需要顺序遍历背包容量，因为我们就是希望用本层更新过的数据来进行递推。

另外，关于先遍历物品还是先遍历背包容量的问题，对于完全背包问题，如果要求的是组合数，即元素之间明确要求没有顺序，那么就外层遍历物品，内层遍历背包容量；如果要求的是排列数，即元素之间明确有顺序要求，那么就外层遍历背包容量，内层遍历物品。

我们假设`coins = [1, 5]`

```java
for (int i = 1; i <= coins.length; i++) { // 遍历物品
    int coin = coins[i - 1];
    for (int j = coin; j <= amount; j++) { // 遍历背包容量
        dp[j] += dp[j - coin];
    }
}
```

这种情况就是先把`1`加入计算，所有`1`的结果计算完了之后，再把`5`加入计算，得到的方案数量只会有`{1, 5}`这种情况，不会出现`{5, 1}`这种情况。

而如果把两个for循环交换顺序：

```java
for (int j = 0; j <= amount; j++) { // 遍历背包容量
    for (int i = 1; i <= coins.length; i++) { // 遍历物品
        int coin = coins[i - 1];
        if (j - coin >= 0) {
            dp[j] += dp[j - coin];
        }
    }
}
```

这种情况就是对于背包容量的每一个值，都是经过了`1`和`5`的计算，包含了`{1, 5}`和`{5, 1}`两种情况。

总结：

- 如果求组合数就是外层for循环遍历物品，内层for遍历背包。
- 如果求排列数就是外层for遍历背包，内层for循环遍历物品。

因此本题的一维数组优化：

```java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            int coin = coins[i - 1];
            for (int j = coin; j <= amount; j++) {
                dp[j] += dp[j - coin];
            }
        }
        return dp[amount];
    }
}
```

时间复杂度：共有`n * amount`个状态需要转移，`O(n * amount)`

### 装满背包有多少种方案？求排列数 [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/description/)

本题同样是求装满背包的方案数，只不过是求排列数，而不是组合数，因此我们可以直接使用上面的结论，外层for遍历背包容量，内层for遍历物品。

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int j = 0; j <= target; j++) {
            for (int i = 1; i <= n; i++) {
                int num = nums[i - 1];
                if (j >= num) {
                    dp[j] += dp[j - num];
                }
            }
        }
        return dp[target];
    }
}
```

### 爬楼梯问题Revisit！

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬`steps[i]`个台阶。你有多少种不同的方法可以爬到楼顶呢？
例1：

```java
输入：n = 3, steps = [1, 2]
输出：3
解释：有3种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 2 阶 + 1 阶
3. 1 阶 + 2 阶
```

例2：

```java
输入：n = 4, steps = [1, 2, 3]
输出：7
解释：有7种方法可以爬到楼顶。
1. [1, 1, 1, 1]
2. [1, 1, 2]
3. [1, 2, 1]
4. [1, 3]
5. [2, 1, 1]
6. [2, 2]
7. [3, 1]
```

根据题意我们可以很容易看出这是一个完全背包问题，并且要求的方案数是排列数（因为先爬1阶再爬2阶和先爬2阶再爬1阶是不同的方案）。

```java
class Solution {
    public int climbStairs(int n, int[] steps) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int j = 0; j <= n; j++) {
            for (int i = 1; i <= steps.length; i++) {
                int steps = steps[i - 1];
                if (j >= step) {
                    dp[j] += dp[j - step];
                }
            }
        }
        return dp[n];
    }
}
```

### 装满背包的最少物品个数？[322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

[题解](https://leetcode.cn/problems/coin-change/solutions/752323/dong-tai-gui-hua-bei-bao-wen-ti-zhan-zai-3265/?orderBy=most_votes)

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            int coin = coins[i - 1];
            for (int j = coin; j <= amount; j++) {
                dp[j] = Math.min(dp[j], dp[j - coin] + 1);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
}
```

### 把问题转换为完全背包问题 [279. 完全平方数](https://leetcode.cn/problems/perfect-squares/description/)

由题意可知：完全平方数就是物品（可以无限件使用），正整数n就是背包容量，问装满这个背包最少有多少物品？

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, n);
        dp[0] = 0;
        for (int i = 1; i * i <= n; i++) {
            for (int j = i * i; j <= n; j++) {
                dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
            }
        }
        return dp[n];
    }
}
```

### 字符串分割？[139. 单词拆分](https://leetcode.cn/problems/word-break/)

我们定义`dp[i]`表示字符串s的前i个字符组成的字符串`s[0...i-1]`能否被拆分成若干个字典中出现的单词。同时我们枚举`s[0...i-1]`中的分割点j，如果`dp[j] == true`并且`s[j...i-1]`也合法的话，那么`dp[i] = true`。
因此，状态转移方程为：`dp[i] = dp[j] && check(s[j...i-1])`
对于边界条件，我们定义`dp[0] = true`表示空串合法，用于递推。

```java
class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                String str = s.substring(j, i);
                if (dp[j] && set.contains(str)) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
```

## 多重背包问题

[4. 多重背包问题 I](https://www.acwing.com/problem/content/4/)

```java
import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int m = sc.nextInt();
        int[] v = new int[n + 1];
        int[] w = new int[n + 1];
        int[] s = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            v[i] = sc.nextInt();
            w[i] = sc.nextInt();
            s[i] = sc.nextInt();
        }
        int[][] f = new int[n + 1][m + 1];
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= s[i] && k * v[i] <= j; k++) {
                    f[i][j] = Math.max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i]);
                }
            }
        }
        System.out.println(f[n][m]);
    }
}
```

# 线性DP

## 最长上升子序列 LIS

[300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)

> 状态转移方程：
> `dp[i] = max(dp[i], dp[j] + 1), for every 0 <= j < i && nums[j] < nums[i]`

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int ans = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            ans = Math.max(ans, dp[i]);
        }
        return ans;
    }
}
```

## 最长公共子序列 LCS

[1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/)

> 状态转移方程：
> `if s1[i-1] == s2[j-1]:`
> `dp[i][j] = dp[i-1][j-1] + 1`
> `else:`
> `dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1])`

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25849280/1678696067332-51c0bcea-cb4f-4f68-bcec-f4dd2de0b12f.png#averageHue=%23f1ede2&clientId=u4b5ba12d-87e8-4&from=paste&height=289&id=ucf49774f&name=image.png&originHeight=1156&originWidth=1540&originalType=binary&ratio=2&rotation=0&showTitle=false&size=215331&status=done&style=none&taskId=uddcbbd97-ed48-4001-b6c3-4a74824c891&title=&width=385)![image.png](https://cdn.nlark.com/yuque/0/2023/png/25849280/1678696111667-410284f6-4e00-412c-8a28-e4bb3aa748c9.png#averageHue=%23e3e7f0&clientId=u4b5ba12d-87e8-4&from=paste&height=288&id=u9d013ab4&name=image.png&originHeight=413&originWidth=464&originalType=binary&ratio=2&rotation=0&showTitle=false&size=14077&status=done&style=none&taskId=uebbc8511-3a88-4c51-b9da-2701849e175&title=&width=324)
需要注意的是`dp[i][j]`是由3个方向转移而来的，但是由于`dp[i-1][j-1]`一定是最小的那个，因此求最大值的时候可以不考虑。

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
}
```

## 编辑距离

[72. 编辑距离](https://leetcode.cn/problems/edit-distance/description/)
`dp[i][j]: 把word1[1...i]变成word2[1...j]需要的最小操作次数`

> 状态转移方程：
> `dp[i][j]会从4种可能转移过来：`
>
> - `word1[i] == word2[j]时，相当于当前不需要操作，即skip`
> - `dp[i-1][j] + 1， 即word1[1...i-1] == word2[1...j]，只要把word1[i]删除即可`
> - `dp[i][j-1] + 1，即word1[1...i] == word2[1...j-1]，只要往word1后面插入word2[j]即可`
> - `dp[i-1][j-1] + 1，即word1[1...i-1] == word2[1...j-1]，只要把word1[i]替换成word2[j]即可`

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25849280/1678698980086-2b4275ad-194b-4b2b-984a-f5572bf93559.png#averageHue=%23fbfbfb&clientId=u4b5ba12d-87e8-4&from=paste&height=720&id=ufd330b28&name=image.png&originHeight=1440&originWidth=2560&originalType=binary&ratio=2&rotation=0&showTitle=false&size=173736&status=done&style=none&taskId=u44385872-75c0-4d2d-b11a-732964447d0&title=&width=1280)

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1]; // skip
                } else {
                    dp[i][j] = min(
                        dp[i - 1][j] + 1, // delete i
                        dp[i][j - 1] + 1, // insert i
                        dp[i - 1][j - 1] + 1 // replace i with j
                    );
                }
            }
        }
        return dp[m][n];
    }

    private int min(int... nums) {
        int min = nums[0];
        for (int x : nums) {
            min = Math.min(min, x);
        }
        return min;
    }
}
```

## 最长回文子序列

[516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/description/) | [题解](https://leetcode.cn/problems/longest-palindromic-subsequence/solutions/67456/zi-xu-lie-wen-ti-tong-yong-si-lu-zui-chang-hui-wen/)
`dp[i][j]`：从下标`i`到下标`j`的最长回文子序列的长度。

> 状态转移方程：
> `dp[i][j]`取决于`dp[i+1][j-1]`和`s[i] == s[j]`。

```java
if (s[i] == s[j])
    // 它俩一定在最长回文子序列中
    dp[i][j] = dp[i + 1][j - 1] + 2;
else
    // s[i+1..j] 和 s[i..j-1] 谁的回文子序列更长？
    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
```

![image.png](https://cdn.nlark.com/yuque/0/2023/png/25849280/1680248953859-75408670-84da-4ae3-98f5-c272b937cf30.png#averageHue=%23f0f2e2&clientId=u0b5a1c6d-0c7b-4&from=paste&height=203&id=u2f80f7b8&name=image.png&originHeight=720&originWidth=1280&originalType=binary&ratio=2&rotation=0&showTitle=false&size=688922&status=done&style=none&taskId=u9431945a-2beb-4fd5-a476-8d81e63d0ea&title=&width=361)![image.png](https://cdn.nlark.com/yuque/0/2023/png/25849280/1680248967758-521a8fd7-b876-42f0-b012-8a648063610b.png#averageHue=%23e8ecd4&clientId=u0b5a1c6d-0c7b-4&from=paste&height=203&id=u3610a2c2&name=image.png&originHeight=720&originWidth=1280&originalType=binary&ratio=2&rotation=0&showTitle=false&size=668159&status=done&style=none&taskId=u2ba0e01c-20da-4889-b61b-816b737e243&title=&width=361)
最终要求`dp[0][n-1]`。对于每个`dp[i][j]`来说，每次计算需要左、下、左下3个位置都已经被计算出来，因此遍历顺序很重要，可以选择反着遍历。

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

另外，本题还有区间DP做法，和反转字符串LCS做法。

# 区间DP

## 最长回文子串

[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/)

```java
class Solution {
    public String longestPalindrome(String s) {
        int n = s.length();
        if(n < 2) return s;
        int l = 0, r = 0, maxLen = 1;
        // dp[i][j] 记录 s[i...j]的子串是否为回文
        boolean[][] dp = new boolean[n][n];
        char[] cs = s.toCharArray();

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (cs[i] == cs[j]) {
                    if (j - i <= 1) dp[i][j] = true; // 回文子串的2种base case：长度为1和长度为2
                    // 倒序遍历顺序决定了dp[i+1][j-1]一定先被计算过了
                    else dp[i][j] = dp[i + 1][j - 1];
                }
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    l = i;
                    r = j;
                }
            }
        }
        return s.substring(l, l + maxLen);
    }
}
```

## 最长回文子序列

[516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/description/) | [题解](https://leetcode.cn/problems/longest-palindromic-subsequence/solutions/930872/gong-shui-san-xie-qu-jian-dp-qiu-jie-zui-h2ya/?orderBy=most_votes)
最长回文子序列问题也可以用区间DP解决。
之所以可以使用区间DP进行求解，是因为在给定一个回文串的基础上，如果在回文串的边缘分别添加2个新的字符，可以通过判断2个字符是否相同来判断新串是否回文。也就是说，**使用小区间的回文状态可以推导出大区间的回文状态值。**

区间DP的基本流程：

- 从小到大枚举区间大小`len`
- 枚举区间左端点`l`，同时根据区间大小和左端点计算出右端点`r = l + len - 1`
- 通过状态转移方程求`dp[l][r]`

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        char[] cs = s.toCharArray();
        int[][] dp = new int[n][n];
        for (int len = 1; len <= n; len++) {
            for (int l = 0; l + len - 1 < n; l++) {
                int r = l + len - 1;
                if (len == 1) {
                    dp[l][r] = 1;
                } else if (len == 2) {
                    dp[l][r] = cs[l] == cs[r] ? 2 : 1;
                } else {
                    dp[l][r] = Math.max(dp[l + 1][r], dp[l][r - 1]);
                    dp[l][r] = Math.max(dp[l][r], dp[l + 1][r - 1] + (cs[l] == cs[r] ? 2 : 0));
                }
            }
        }
        return dp[0][n - 1];
    }
}
```

## 石子合并

[282. 石子合并](https://www.acwing.com/problem/content/284/) | [题解](https://www.acwing.com/solution/content/13945/)

```java
import java.util.*;

class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] stones = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            stones[i] = sc.nextInt();
        }
        int[] s = new int[n + 1]; // 前缀和数组
        for (int i = 1; i <= n; i++) {
            s[i] = s[i - 1] + stones[i];
        }
        int[][] dp = new int[n + 1][n + 1];
        for (int[] e : dp) Arrays.fill(e, Integer.MAX_VALUE); // 初始化
        for (int len = 1; len <= n; len++) { // 区间 DP 枚举套路：长度+左端点
            for (int i = 1; i + len - 1 <= n; i++) {
                int l = i, r = i + len - 1; // 左端点和右端点
                if (len == 1) {
                    dp[l][r] = 0; // 初始化
                    continue;
                }
                for (int k = l; k < r; k++) { // 枚举分割点
                    dp[l][r] = Math.min(dp[l][r], dp[l][k] + dp[k + 1][r] + s[r] - s[l - 1]);
                }
            }
        }
        System.out.println(dp[1][n]);
    }
}
```

```java
for (int len = 1; len <= n; len++) {         // 区间长度
    for (int i = 1; i + len - 1 <= n; i++) { // 枚举起点
        int j = i + len - 1;                 // 区间终点
        if (len == 1) {
            dp[i][j] = 初始值
            continue;
        }

        for (int k = i; k < j; k++) {        // 枚举分割点，构造状态转移方程
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + w[i][j]);
        }
    }
}
```