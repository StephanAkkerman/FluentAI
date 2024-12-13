import Image from "next/image";
import logo from "../../public/logo.png";

export default function Header() {
  return (
    <header className="w-full flex items-center justify-between p-4 border-b border-gray-200">
      <div className="flex items-center space-x-2">
        <Image src={logo} alt="FluentAI Logo" width={40} height={40} />
        <h1 className="text-xl font-bold">FluentAI</h1>
      </div>
      {/* Placeholder for settings icon or nav links */}
      <div>
        {/* Settings button or gear icon will go here in the future */}
      </div>
    </header>
  );
}
