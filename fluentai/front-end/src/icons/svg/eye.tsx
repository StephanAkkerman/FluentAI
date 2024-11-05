import React from "react";

type EyeSvgProps = {
  className: string;
  onClick?: () => void;
};

const EyeSvg: React.FC<EyeSvgProps> = ({ className, onClick }) => {
  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  return (
    <div className={className} onClick={handleClick}>
      <svg viewBox="0 0 24 24">
        <path
          d="M3 14C3 9.02944 7.02944 5 12 5C16.9706 5 21 9.02944 21 14M17 14C17 16.7614 14.7614 19 12 19C9.23858 19 7 16.7614 7 14C7 11.2386 9.23858 9 12 9C14.7614 9 17 11.2386 17 14Z"
          strokeWidth="2"
        ></path>
      </svg>
    </div>
  );
};

export default EyeSvg;
