### compile

cd BstarTreeFP/src 

make or cmake.build

exe under build/fp

### debug

for conda, replace conda link file to system link file

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

### execute

in qsbm

run \<exe> \<data set> \<output path> \<dead space ratio> \<mode> \<submode>

example: ./build/fp ./data/gsrc/n100 ./tools/BstarTreeFP/output/ 0.1 SB sparse

### SA result

<table>
  <tr>
    <th></th>
    <th colspan="3">wirelength</th>
    <th colspan="3">runtime (s)</th>
  </tr>
  <tr>
    <td></td>
    <td>n100</td>
    <td>n200</td>
    <td>n300</td>
    <td>n100</td>
    <td>n200</td>
    <td>n300</td>
  </tr>
  <tr>
    <td >0.1 </td>
    <td >239059 </td>
    <td >411620 </td>
    <td >568221 </td>
    <td >25.01 </td>
    <td >100.27 </td>
    <td >225.14 </td>
  </tr>
  <tr>
    <td >0.15 </td>
    <td >224626 </td>
    <td >415209 </td>
    <td >563408 </td>
    <td >25.03 </td>
    <td >100.24 </td>
    <td >225.46 </td>
  </tr>
</table>

### SB result
