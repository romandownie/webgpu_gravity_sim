async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!device) {
    fail('need a browser that supports WebGPU');
    return;
  }

  // Get a WebGPU context from the canvas and configure it
  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
  });

  const kNumPoints = 30000;
  const scale = 10000000;
  const startDate = new Date();
  const dateStart = startDate.getTime();

  // Vertex and Fragment Shaders
  const shaderModule = device.createShaderModule({
    code: `
      struct Vertex {
        @location(0) position: vec2f,
        @location(1) vel: vec2f,
        @location(2) mass: f32,
        @location(3) filler: f32,
      };

      struct VSOutput {
        @builtin(position) position: vec4f,
        @location(4) vel: vec2f,
        @location(5) mass: f32,
      };

      @vertex fn vs(vert: Vertex) -> VSOutput {
        var vsOut: VSOutput;
        vsOut.position = vec4f(vert.position/vec2f(${scale}.0), 0, 1);
        vsOut.vel = vert.vel;
        vsOut.mass = vert.mass;
        return vsOut;
      }

      @fragment fn fs(vsOut: VSOutput) -> @location(0) vec4f {
        return vec4f(vsOut.mass/ pow(2, 20.0), vsOut.vel.x/((${scale})/8.0), vsOut.vel.y/((${scale})/8.0), 1); // yellow
      }
    `,
  });

  // Compute Shader
  const computeShaderModule = device.createShaderModule({
    code: `
      struct particle {
        pos: vec2f,
        vel: vec2f,
        mass: f32,
        filler: f32,
      };  
    
      @group(0) @binding(0) var<storage, read_write> vertices: array<particle, ${kNumPoints}>;
      @group(0) @binding(1) var<uniform> deltaTime: f32;
      @group(0) @binding(2) var<uniform> ms: f32;


      // from https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
      // A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
      fn hash(x: u32) -> u32 {
        var x_var = x;
        x_var = x_var + (x_var << 10u);
        x_var = x_var ^ (x_var >> 6u);
        x_var = x_var + (x_var << 3u);
        x_var = x_var ^ (x_var >> 11u);
        x_var = x_var + (x_var << 15u);
        return x_var;
      } 
      
      // Compound versions of the hashing algorithm I whipped together.
      fn hash1(v: vec2<u32>) -> u32 {
        return hash(v.x ^ hash(v.y));
      } 
      
      fn hash2(v: vec3<u32>) -> u32 {
        return hash(v.x ^ (hash(v.y) ^ hash(v.z)));
      } 
      
      fn hash3(v: vec4<u32>) -> u32 {
        return hash(v.x ^ (hash(v.y) ^ (hash(v.z) ^ hash(v.w))));
      } 
      
      // Construct a float with half-open range [0:1] using low 23 bits.
      // All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
      fn floatConstruct(m: u32) -> f32 {
        var m_var = m;
        let ieeeMantissa: u32 = 8388607u; // binary32 mantissa bitmask
        let ieeeOne: u32 = 1065353216u; // 1.0 in IEEE binary32
        m_var = m_var & (ieeeMantissa); // Keep only mantissa bits (fractional part)
        m_var = m_var | (ieeeOne); // Add fractional part to 1.0
        let f: f32 = bitcast<f32>(m_var); // Range [1:2]
        return f - 1.0; // Range [0:1]
      } 
      
      // Pseudo-random value in half-open range [0:1].
      fn rand(x: f32) -> f32 {
        return floatConstruct(hash(bitcast<u32>(x)));
      } 
      
      fn rand1(v: vec2<f32>) -> f32 {
        return floatConstruct(hash1(vec2u(bitcast<u32>(v.x), bitcast<u32>(v.y))));
      } 
      
      fn rand2(v: vec3<f32>) -> f32 {
        return floatConstruct(hash2(  vec3u( bitcast<u32>(v.x), bitcast<u32>(v.y), bitcast<u32>(v.z)   )));
      } 
      
      fn rand3(v: vec4<f32>) -> f32 {
        return floatConstruct(hash3(   vec4u( bitcast<u32>(v.x), bitcast<u32>(v.y), bitcast<u32>(v.z), bitcast<u32>(v.w)   )));
      } 
      
      // end rand



      const G = 0.00000000006667; //maybe not precise enough 0.00000000006667

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3u) {
        let index = id.x;
        if (index >= ${kNumPoints}u) { return; }
        // Modify vertex position (example: move right with animation)
        //vertices[index] = vertices[index] + vec2f(deltaTime, 0.0);

        let arraySize = i32(${kNumPoints});
        let point = vertices[index];
        var posUpdate = vec2f(0.0);
        var velUpdate = vec2f(0.0);

        if(point.mass > pow(2, 20.0)) {
          vertices[index].mass = pow(2, 20.0);
          //vertices[index].pos = vec2f(rand(f32(index))*${scale}*2.0 - ${scale}, rand(f32(index))*${scale}*2.0 - ${scale});
          //vertices[index].vel = vec2f(rand(f32(index))*${scale}*2.0 - ${scale}, rand(f32(index))*${scale}*2.0 - ${scale});
        }

        for(var i = 0; i < arraySize; i++) {
          if (i == i32(index)) {
            continue;
          }
          // calculate gravity force F = Gm1m2/r^2
          let otherPoint = vertices[i];
          var r = sqrt(pow((point.pos.x - otherPoint.pos.x), 2.0) + pow((point.pos.y - otherPoint.pos.y), 2.0));
          if (r < 0.1) { // handles weird jumping when too close
            //create new point and then eat the mass of the old one
            if (point.mass >= pow(2, 9.5) && otherPoint.mass >= pow(2, 9.5)) {
              // do nothing and do the math between them
              break;
            }
            if (point.mass >= (otherPoint.mass - 1.0)) {
              vertices[index].mass += otherPoint.mass;
              vertices[i].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0 + 10.0;
              vertices[i].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
              vertices[i].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
            } else {
              vertices[i].mass += otherPoint.mass;
              vertices[index].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0 + 10.0;
              vertices[index].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
              vertices[index].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
            }
            break;
          } 
          // if (r > ${scale}*(pow(2.0, 20.0)/point.mass)*0.00018 && r > ${scale}*(pow(2.0, 20.0)/otherPoint.mass)*0.00018 ) { // 0.28 works really well
          //   continue;
          // }
          if (r > ${scale}*0.28) { // 0.28 works really well
            continue;
          }
          let F = G*point.mass*otherPoint.mass/pow(r, 2.0);
          // F/m = a
          let a = F/point.mass;
          // apply offset based on acceleration, vit + 1/2at^2
            //ignore for now because now way to track vi
          let time = deltaTime / 100.0; //4.0 works really well
          let v = vec2f(   (point.vel*time + vec2f(0.5*a*pow(time, 2.0)))  );
          posUpdate += normalize(otherPoint.pos - point.pos) * v;
          velUpdate += v;
        }

        var newPos = vertices[index].pos + posUpdate;
        
        // if (newPos.x > ${scale}) {
        //   newPos = vertices[index].pos - posUpdate;
        //   vertices[index].vel.x -= vertices[index].vel.x * 2.0 - ${scale} * rand2(vec3f(id));
        // }
        // if (newPos.x < -${scale}) {
        //   newPos = vertices[index].pos - posUpdate;
        //   vertices[index].vel.x -= vertices[index].vel.x * 2.0 - ${scale} * rand2(vec3f(id));
        // }
        // if (newPos.y > ${scale}) {
        //   newPos = vertices[index].pos - posUpdate;
        //   vertices[index].vel.y -= vertices[index].vel.y * 2.0 - ${scale} * rand2(vec3f(id));
        // }
        // if (newPos.y < -${scale}) {
        //   newPos = vertices[index].pos - posUpdate;
        //   vertices[index].vel.y -= vertices[index].vel.y * 2.0 - ${scale} * rand2(vec3f(id));
        // }

        if (newPos.x > ${scale}) {
          vertices[index].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0+ 10.0;
          vertices[index].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          vertices[index].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          newPos = vertices[index].pos;
        }
        if (newPos.x < -${scale}) {
          vertices[index].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0 + 10.0;
          vertices[index].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          vertices[index].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          newPos = vertices[index].pos;
        }
        if (newPos.y > ${scale}) {
          vertices[index].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0 + 10.0;
          vertices[index].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          vertices[index].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          newPos = vertices[index].pos;
        }
        if (newPos.y < -${scale}) {
          vertices[index].mass = rand2(vec3f(f32(id.x) + f32(index) + ms))*100.0 + 10.0;
          vertices[index].pos = vec2f(rand1(vec2f(ms*0.26758,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*1.234,f32(id.x + id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          vertices[index].vel = vec2f(rand1(vec2f(ms*1.236324011,f32(id.x/${kNumPoints})))*${scale}*2.0 - ${scale}, rand1(vec2f(ms*0.13647,f32(id.y/${kNumPoints})))*${scale}*2.0 - ${scale});
          newPos = vertices[index].pos;
        }

        vertices[index].pos = (newPos);
        vertices[index].vel = max((vertices[index].vel + velUpdate),-vec2f(1000000.0)); // for stability
        vertices[index].vel = min((vertices[index].vel + velUpdate),vec2f(1000000.0)); // for stability
        //vertices[index].pos = vertices[index].pos + vec2f(vertices[index].vel.x/100.0, vertices[index].vel.y/100.0);
      }
    `,
  });

  const deltaTimeBuffer = device.createBuffer({
    label: 'Delta Time Buffer',
    size: 4, // sizeof(f32)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const msBuffer = device.createBuffer({
    label: 'millisecond Time Buffer',
    size: 4, // sizeof(f32)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const vertexData = new Float32Array(kNumPoints * 6); //number here is number of floats in each particle
  const rand = (min, max) => min + Math.random() * (max - min);
  for (let i = 0; i < kNumPoints; ++i) {
    var seed;
    if (i%1000 == 0){
      seed = 100000.0;
    } else {
      seed = 1.0;
    }
    
    const offset = i * 6;
    vertexData[offset + 0] = rand(-scale, scale); // x
    vertexData[offset + 1] = rand(-scale, scale); // y
    vertexData[offset + 2] = rand(-100, 100); // velocity x
    vertexData[offset + 3] = rand(-100, 100); // velocity y
    vertexData[offset + 4] = rand(50.0, 100.0*seed); // mass
    vertexData[offset + 5] = rand(0.00001, scale); // filler
  }

  const vertexBuffer = device.createBuffer({
    label: 'vertex buffer vertices',
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: vertexBuffer },
      },
      {
        binding: 1,
        resource: { buffer: deltaTimeBuffer },
      },
      {
        binding: 2,
        resource: { buffer: msBuffer },
      },
    ],
  });

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: 'main',
    },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs',
      buffers: [
        {
          arrayStride: 6 * 4, // 5 floats, 4 bytes each
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },  // position
            {shaderLocation: 1, offset: 2*4, format: 'float32x2'},
            {shaderLocation: 2, offset: 4*4, format: 'float32'},
            {shaderLocation: 3, offset: 5*4, format: 'float32'},
          ],
        },
      ],
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs',
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: undefined, // to be filled out when we render
        clearValue: [0.3, 0.3, 0.3, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  };

  var pastTime = dateStart;
  function render() {
    const encoder = device.createCommandEncoder();
  
    // Update deltaTime uniform
    const date = new Date();
    let currTime = date.getTime();
    const deltaTime = (currTime - pastTime) * 0.0001; // example value, adjust as needed, this value is in miliseconds
    //console.log(deltaTime);
    pastTime = currTime;
    const deltaTimeArray = new Float32Array([deltaTime]);
    const msTimeArray = new Float32Array([date.getMilliseconds()]);
    device.queue.writeBuffer(deltaTimeBuffer, 0, deltaTimeArray);
    device.queue.writeBuffer(msBuffer, 0, msTimeArray);
  
    // Get the current texture from the canvas context
    const canvasTexture = context.getCurrentTexture();
    if (!canvasTexture) return; // Return if the texture is not available yet
  
    // Configure render pass descriptor with the current texture view
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();
  
    // Compute pass to update vertex positions
    {
      const computePass = encoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(Math.ceil(kNumPoints / 64));
      computePass.end();
    }
  
    // Render pass
    {
      const pass = encoder.beginRenderPass(renderPassDescriptor);
      pass.setPipeline(renderPipeline);
      pass.setVertexBuffer(0, vertexBuffer);
      pass.draw(kNumPoints);
      pass.end();
    }
  
    // Finish encoding commands and submit
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
  }
  

  const observer = new ResizeObserver(entries => {
    for (const entry of entries) {
      const canvas = entry.target;
      const width = entry.contentBoxSize[0].inlineSize;
      const height = entry.contentBoxSize[0].blockSize;
      canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
      canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
      renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
      render();
    }
  });
  observer.observe(canvas);

  setInterval(render, 8); // Update every 16 milliseconds (about 60 frames per second)
}

function fail(msg) {
  alert(msg);
}

main();
