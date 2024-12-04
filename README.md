[comment]: <> 

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> Get a dense Depth map from sparse LiDAR point clouds
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="">Blog</a> 
  | <a href="https://github.com/BerensRWU/DenseMap">Original Github Page</a>
  </h3>
  <div align="justify">
  The function gets the projected LiDAR point cloud, the size of the camera image and the grid size. 
  A grid size of 4 means a 9x9 neighbourhood is used and weighted depth information is calculated according to the distance of the neighbourhood.
  </div>



<!-- 下面是初始化并创建github仓库 -->
<!-- 
~~~
#进入到该项目Terminal窗口,执行删除.git目录
rm -rf .git
 
#此时项目已经不再被git版本库所管理,就可以创建忽略文件了;
创建.gitignore文件
 
#然后重新初始化该项目,该项目又受git版本控制了;
git init
 
#然后进行add了,将所有的项目都提交到缓存
git add .
 
#然后提交到git本地仓库中
git commit -m "提交初始化版本"
 
#在gitee网站中创建一个仓库,进行与远程仓库关联
git remote add origin git@****/*.git
 
#然后推送到远程仓库
git push origin master
 
#在重新执行下推送
git push -u origin master
~~~ -->
