# Argmax（最大值下标）

> reduce 的加强版：归约的不是单个值，而是 (value, index) 对，比较时连下标一起移动。

## 含义
```text
max(x)    → 最大值
argmax(x) → 最大值的下标
平局规则：取最小下标（多个相同最大值时返回第一个）
```

## 实现要点
```text
1. 归约单元变成 struct { float val; int idx; }
2. 比较：if (b.val > a.val) a = b;  否则保留 a（平局取小下标）
3. warp shuffle：val 和 idx 各 shuffle 一次（一次只能传 32bit）
4. 两级归约结构同 reduce：warp 内 + warp 间(shared)
5. 两趟 kernel：第一趟每 block 出一个 (val,idx)，第二趟合并
```

## 完成标准
```text
[ ] CPU reference 校验（值和下标都对，平局取小下标）
[ ] N = 1<<20 等规模
[ ] 处理负数、重复最大值
[ ] 一段口述：argmax 和 reduce 的区别
```
