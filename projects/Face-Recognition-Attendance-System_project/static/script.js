// Initialize particles.js
particlesJS('particles-js', {
    "particles": {
      "number": {
        "value": 100,
        "density": {
          "enable": true,
          "value_area": 800
        }
      },
      "color": {
        "value": "#ffffff"
      },
      "shape": {
        "type": "circle",
        "stroke": {
          "width": 0,
          "color": "#000000"
        },
        "polygon": {
          "nb_sides": 5
        }
      },
      "opacity": {
        "value": 0.5,
        "random": false,
        "anim": {
          "enable": false,
          "speed": 1,
          "opacity_min": 0.1,
          "sync": false
        }
      },
      "size": {
        "value": 5,
        "random": true,
        "anim": {
          "enable": false,
          "speed": 40,
          "size_min": 0.1,
          "sync": false
        }
      },
      "line_linked": {
        "enable": true,
        "distance": 150,
        "color": "#ffffff",
        "opacity": 0.4,
        "width": 1
      },
      "move": {
        "enable": true,
        "speed": 3,
        "direction": "none",
        "random": false,
        "straight": false,
        "out_mode": "out",
        "bounce": false,
        "attract": {
          "enable": false,
          "rotateX": 600,
          "rotateY": 1200
        }
      }
    },
    "interactivity": {
      "detect_on": "canvas",
      "events": {
        "onhover": {
          "enable": true,
          "mode": "grab"
        },
        "onclick": {
          "enable": true,
          "mode": "push"
        },
        "resize": true
      },
      "modes": {
        "grab": {
          "distance": 200,
          "line_linked": {
            "opacity": 1
          }
        },
        "bubble": {
          "distance": 400,
          "size": 40,
          "duration": 2,
          "opacity": 8,
          "speed": 3
        },
        "repulse": {
          "distance": 200,
          "duration": 0.4
        },
        "push": {
          "particles_nb": 4
        },
        "remove": {
          "particles_nb": 2
        }
      }
    },
    "retina_detect": true
  });
  
  // Div glow effect on hover
  const hoverDiv = document.querySelector('.container, .containeradd, .containerattendance');
  
  hoverDiv.addEventListener('mousemove', (e) => {
    const rect = hoverDiv.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
  
    const glowSize = 20;
    const offsetX = (x / hoverDiv.offsetWidth - 0.5) * glowSize;
    const offsetY = (y / hoverDiv.offsetHeight - 0.5) * glowSize;
  
    const rotateX = ((y / hoverDiv.offsetHeight) - 0.5) * 10;
    const rotateY = ((x / hoverDiv.offsetWidth) - 0.5) * -20;
  
    hoverDiv.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    hoverDiv.style.boxShadow = `${offsetX}px ${offsetY}px 15px rgba(255, 255, 255, 0.8), ${offsetX * 1.5}px ${offsetY * 1.5}px 30px rgba(255, 255, 255, 0.4)`;
    hoverDiv.style.background = `radial-gradient(circle at ${x}px ${y}px, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0) 80%)`;
  });
  
  hoverDiv.addEventListener('mouseleave', () => {
    hoverDiv.style.transform = 'rotateX(0deg) rotateY(0deg)';
    hoverDiv.style.boxShadow = 'none';
    hoverDiv.style.background = 'rgba(255, 255, 255, 0.1)';
  });
  