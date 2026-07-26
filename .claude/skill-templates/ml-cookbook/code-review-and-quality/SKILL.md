---
name: "code-review-and-quality"
description: "代码审查与质量保障 — 五维审查（安全性、正确性、性能、可维护性、规范对齐）。当需要审查代码、检查质量、验证实现是否符合规范或查找 bug 时使用。"
---

# Code Review & Quality Skill

代码审查与质量保障，适用于本项目（React + TypeScript + CloudBase + 高德地图）。

## 审查维度

### 1. 安全性（权重 25%）
- OWASP Top 10 检查（XSS、CSRF、注入）
- CloudBase Publishable Key 不可暴露 admin 权限
- 高德 Key 使用安全密钥（securityJsCode）配置
- localStorage 不存储敏感数据
- API 调用参数校验

### 2. 正确性（权重 25%）
- TypeScript 类型安全（strict mode）
- CloudBase 数据库操作是否符合安全规则
- watch() 订阅的 cleanup 是否正确
- 组件卸载时是否取消异步操作（避免 setState on unmounted）
- 乐观更新和真实数据之间的最终一致性

### 3. 性能（权重 15%）
- React 重渲染优化（useMemo、useCallback、React.memo）
- 高德地图实例生命周期管理（避免内存泄漏）
- watch() 订阅数量控制（每个城市最多 4 个）
- 搜索防抖（300ms）和中文输入法兼容
- 虚拟列表优化长标记列表
- 字体加载优化（中文字体子集内嵌）

### 4. 可维护性（权重 15%）
- 组件职责单一，不超过 200 行
- Hook 可复用，逻辑与 UI 分离
- CloudBase 操作集中在 lib/ 层，不在组件中直接调 db
- 高德 API 调用集中在 useAMap Hook 中
- 类型定义集中在 types/ 目录

### 5. 规范对齐（权重 20%）
- 代码实现是否与规范文档 `.spec.md` 一致
- 数据库字段名和类型是否匹配
- 路由设计是否符合规范
- Context/State 结构是否与设计一致

## 审查流程

1. **逐文件审查** → 标记问题（Critical/Major/Minor）
2. **跨文件检查** → 依赖关系、导入路径、Context 提供者嵌套
3. **构建检查** → `npm run build` 无 TypeScript 错误
4. **规范对齐** → 对照 `.spec.md` 检查是否存在功能遗漏

## 常用检查项

- [ ] 所有 Component 的 useEffect 是否有 cleanup
- [ ] 所有 watch() 调用是否有 close()
- [ ] 异步操作是否处理了 loading/error 状态
- [ ] TypeScript 无 any 类型（除高德 API 回调）
- [ ] 无 console.log 残留
- [ ] 错误边界覆盖关键组件
- [ ] 移动端 touch-action 冲突处理
- [ ] 微信内置浏览器兼容性（es2015 target）
- [ ] CloudBase 安全域名是否包含当前访问 origin
